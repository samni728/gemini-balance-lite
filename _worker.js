export default {
  async fetch(request, env, ctx) {
    try {
      const url = new URL(request.url);
      const pathname = url.pathname;

      // CORS preflight
      if (request.method === "OPTIONS") {
        return new Response(null, {
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
          },
        });
      }

      // Health check
      if (pathname === "/" || pathname === "/index.html") {
        return new Response(
          "Proxy is Running!  More Details: https://github.com/tech-shrimp/gemini-balance-lite",
          {
            status: 200,
            headers: corsHeaders({ "Content-Type": "text/html" }),
          }
        );
      }

      // Verify API keys (SSE)
      if (pathname === "/verify" && request.method === "POST") {
        return handleVerification(request);
      }

      // OpenAI-compatible routes
      if (
        pathname.endsWith("/chat/completions") ||
        pathname.endsWith("/completions") ||
        pathname.endsWith("/embeddings") ||
        pathname.endsWith("/models")
      ) {
        return openaiAdapterFetch(request);
      }

      // Default: Gemini native proxy
      return proxyToGemini(request, url);
    } catch (err) {
      console.error(err);
      return new Response(
        err?.message || "Internal Error",
        corsHeaders({ status: 500 })
      );
    }
  },
};

function corsHeaders(init) {
  const headers = new Headers(init?.headers);
  headers.set("Access-Control-Allow-Origin", "*");
  return { ...init, headers };
}

async function proxyToGemini(request, url) {
  const { pathname, search } = url;
  const targetUrl = `https://generativelanguage.googleapis.com${pathname}${search}`;

  const headers = new Headers();
  for (const [key, value] of request.headers.entries()) {
    const lowerKey = key.trim().toLowerCase();
    if (lowerKey === "x-goog-api-key") {
      const selected = selectOneFromCommaList(value);
      if (selected) {
        console.log(`Gemini Selected API Key: ${maskKey(selected)}`);
        headers.set("x-goog-api-key", selected);
      }
    } else if (lowerKey === "content-type") {
      headers.set(key, value);
    }
  }

  const response = await fetch(targetUrl, {
    method: request.method,
    headers,
    body: request.body,
  });

  const responseHeaders = new Headers(response.headers);
  responseHeaders.delete("transfer-encoding");
  responseHeaders.delete("connection");
  responseHeaders.delete("keep-alive");
  responseHeaders.delete("content-encoding");
  responseHeaders.set("Referrer-Policy", "no-referrer");
  responseHeaders.set("Access-Control-Allow-Origin", "*");

  return new Response(response.body, {
    status: response.status,
    headers: responseHeaders,
  });
}

function selectOneFromCommaList(value) {
  const list = String(value)
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  if (list.length === 0) return undefined;
  return list[Math.floor(Math.random() * list.length)];
}

function maskKey(key) {
  if (!key || key.length < 14) return key;
  return `${key.slice(0, 7)}......${key.slice(-7)}`;
}

// --------------------------- Verify Keys (SSE) ---------------------------
async function verifyKey(key, controller) {
  const url =
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent";
  const body = {
    contents: [{ role: "user", parts: [{ text: "Hello" }] }],
  };
  let result;
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-goog-api-key": key,
      },
      body: JSON.stringify(body),
    });
    if (response.ok) {
      await response.text();
      result = { key: maskKey(key), status: "GOOD" };
    } else {
      const errorData = await response
        .json()
        .catch(() => ({ error: { message: "Unknown error" } }));
      result = {
        key: maskKey(key),
        status: "BAD",
        error: errorData.error?.message,
      };
    }
  } catch (e) {
    result = { key: maskKey(key), status: "ERROR", error: e.message };
  }
  controller.enqueue(
    new TextEncoder().encode("data: " + JSON.stringify(result) + "\n\n")
  );
}

async function handleVerification(request) {
  try {
    const authHeader = request.headers.get("x-goog-api-key");
    if (!authHeader) {
      return new Response(
        JSON.stringify({ error: "Missing x-goog-api-key header." }),
        corsHeaders({
          status: 400,
          headers: { "Content-Type": "application/json" },
        })
      );
    }
    const keys = authHeader
      .split(",")
      .map((k) => k.trim())
      .filter(Boolean);

    const stream = new ReadableStream({
      async start(controller) {
        const tasks = keys.map((key) => verifyKey(key, controller));
        await Promise.all(tasks);
        controller.close();
      },
    });

    return new Response(
      stream,
      corsHeaders({
        status: 200,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      })
    );
  } catch (e) {
    return new Response(
      JSON.stringify({ error: "An unexpected error occurred: " + e.message }),
      corsHeaders({
        status: 500,
        headers: { "Content-Type": "application/json" },
      })
    );
  }
}

// ------------------------ OpenAI-compatible Adapter ------------------------
const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";
const API_CLIENT = "genai-js/0.21.0";

function makeHeaders(apiKey, more) {
  return {
    "x-goog-api-client": API_CLIENT,
    ...(apiKey && { "x-goog-api-key": apiKey }),
    ...more,
  };
}

function fixCors(init) {
  return corsHeaders(init || {});
}

function toBase64(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode.apply(null, chunk);
  }
  return btoa(binary);
}

async function parseImg(url) {
  let mimeType, data;
  if (url.startsWith("http://") || url.startsWith("https://")) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText} (${url})`);
    }
    mimeType = response.headers.get("content-type");
    data = toBase64(await response.arrayBuffer());
  } else {
    const match = url.match(/^data:(?<mimeType>.*?)(;base64)?,(?<data>.*)$/);
    if (!match) {
      throw new HttpError("Invalid image data: " + url, 400);
    }
    ({ mimeType, data } = match.groups);
  }
  return { inlineData: { mimeType, data } };
}

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

function transformCandidates(key, cand) {
  const SEP = "\n\n|>";
  const message = { role: "assistant", content: [] };
  for (const part of cand.content?.parts ?? []) {
    if (part.functionCall) {
      const fc = part.functionCall;
      message.tool_calls = message.tool_calls ?? [];
      message.tool_calls.push({
        id: fc.id ?? "call_" + generateId(),
        type: "function",
        function: { name: fc.name, arguments: JSON.stringify(fc.args) },
      });
    } else {
      message.content.push(part.text);
    }
  }
  message.content = message.content.join(SEP) || null;
  return {
    index: cand.index || 0,
    [key]: message,
    logprobs: null,
    finish_reason: message.tool_calls
      ? "tool_calls"
      : reasonsMap[cand.finishReason] || cand.finishReason,
  };
}

const reasonsMap = {
  STOP: "stop",
  MAX_TOKENS: "length",
  SAFETY: "content_filter",
  RECITATION: "content_filter",
};

function transformUsage(data) {
  return {
    completion_tokens: data?.candidatesTokenCount,
    prompt_tokens: data?.promptTokenCount,
    total_tokens: data?.totalTokenCount,
  };
}

function generateId() {
  const characters =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const randomChar = () =>
    characters[Math.floor(Math.random() * characters.length)];
  return Array.from({ length: 29 }, randomChar).join("");
}

function adjustProps(schemaPart) {
  if (typeof schemaPart !== "object" || schemaPart === null) return;
  if (Array.isArray(schemaPart)) {
    schemaPart.forEach(adjustProps);
    return;
  }
  if (
    schemaPart.type === "object" &&
    schemaPart.properties &&
    schemaPart.additionalProperties === false
  ) {
    delete schemaPart.additionalProperties;
  }
  Object.values(schemaPart).forEach(adjustProps);
}

function adjustSchema(schema) {
  const obj = schema[schema.type];
  delete obj?.strict;
  return adjustProps(schema);
}

const harmCategory = [
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_CIVIC_INTEGRITY",
];
const safetySettings = harmCategory.map((category) => ({
  category,
  threshold: "BLOCK_NONE",
}));

const fieldsMap = {
  frequency_penalty: "frequencyPenalty",
  max_completion_tokens: "maxOutputTokens",
  max_tokens: "maxOutputTokens",
  n: "candidateCount",
  presence_penalty: "presencePenalty",
  seed: "seed",
  stop: "stopSequences",
  temperature: "temperature",
  top_k: "topK",
  top_p: "topP",
};
const thinkingBudgetMap = { low: 1024, medium: 8192, high: 24576 };

function transformConfig(req) {
  const cfg = {};
  for (const key in req) {
    const mapped = fieldsMap[key];
    if (mapped) cfg[mapped] = req[key];
  }
  if (req.response_format) {
    switch (req.response_format.type) {
      case "json_schema":
        adjustSchema(req.response_format);
        cfg.responseSchema = req.response_format.json_schema?.schema;
        if (cfg.responseSchema && "enum" in cfg.responseSchema) {
          cfg.responseMimeType = "text/x.enum";
          break;
        }
      case "json_object":
        cfg.responseMimeType = "application/json";
        break;
      case "text":
        cfg.responseMimeType = "text/plain";
        break;
      default:
        throw new HttpError("Unsupported response_format.type", 400);
    }
  }
  if (req.reasoning_effort) {
    cfg.thinkingConfig = {
      thinkingBudget: thinkingBudgetMap[req.reasoning_effort],
    };
  }
  return cfg;
}

async function transformMsg({ content }) {
  const parts = [];
  if (!Array.isArray(content)) {
    parts.push({ text: content });
    return parts;
  }
  for (const item of content) {
    switch (item.type) {
      case "text":
        parts.push({ text: item.text });
        break;
      case "image_url":
        parts.push(await parseImg(item.image_url.url));
        break;
      case "input_audio":
        parts.push({
          inlineData: {
            mimeType: "audio/" + item.input_audio.format,
            data: item.input_audio.data,
          },
        });
        break;
      default:
        throw new HttpError(`Unknown "content" item type: "${item.type}"`, 400);
    }
  }
  if (content.every((item) => item.type === "image_url")) {
    parts.push({ text: "" });
  }
  return parts;
}

function transformFnResponse({ content, tool_call_id }, parts) {
  if (!parts.calls)
    throw new HttpError("No function calls found in the previous message", 400);
  let response;
  try {
    response = JSON.parse(content);
  } catch (e) {
    throw new HttpError("Invalid function response: " + content, 400);
  }
  if (
    typeof response !== "object" ||
    response === null ||
    Array.isArray(response)
  ) {
    response = { result: response };
  }
  if (!tool_call_id) throw new HttpError("tool_call_id not specified", 400);
  const { i, name } = parts.calls[tool_call_id] ?? {};
  if (!name) throw new HttpError("Unknown tool_call_id: " + tool_call_id, 400);
  if (parts[i])
    throw new HttpError("Duplicated tool_call_id: " + tool_call_id, 400);
  parts[i] = {
    functionResponse: {
      id: tool_call_id.startsWith("call_") ? null : tool_call_id,
      name,
      response,
    },
  };
}

function transformFnCalls({ tool_calls }) {
  const calls = {};
  const parts = tool_calls.map(
    ({ function: { arguments: argstr, name }, id, type }, i) => {
      if (type !== "function")
        throw new HttpError(`Unsupported tool_call type: "${type}"`, 400);
      let args;
      try {
        args = JSON.parse(argstr);
      } catch {
        throw new HttpError("Invalid function arguments: " + argstr, 400);
      }
      calls[id] = { i, name };
      return {
        functionCall: { id: id.startsWith("call_") ? null : id, name, args },
      };
    }
  );
  parts.calls = calls;
  return parts;
}

async function transformMessages(messages) {
  if (!messages) return;
  const contents = [];
  let system_instruction;
  for (const item of messages) {
    switch (item.role) {
      case "system":
        system_instruction = { parts: await transformMsg(item) };
        continue;
      case "tool": {
        let { role, parts } = contents[contents.length - 1] ?? {};
        if (role !== "function") {
          const calls = parts?.calls;
          parts = [];
          parts.calls = calls;
          contents.push({ role: "function", parts });
        }
        transformFnResponse(item, parts);
        continue;
      }
      case "assistant":
        item.role = "model";
        break;
      case "user":
        break;
      default:
        throw new HttpError(`Unknown message role: "${item.role}"`, 400);
    }
    contents.push({
      role: item.role,
      parts: item.tool_calls
        ? transformFnCalls(item)
        : await transformMsg(item),
    });
  }
  if (system_instruction) {
    if (!contents[0]?.parts.some((part) => part.text)) {
      contents.unshift({ role: "user", parts: { text: " " } });
    }
  }
  return { system_instruction, contents };
}

function transformTools(req) {
  let tools, tool_config;
  if (req.tools) {
    const funcs = req.tools.filter(
      (tool) =>
        tool.type === "function" && tool.function?.name !== "googleSearch"
    );
    if (funcs.length > 0) {
      funcs.forEach(adjustSchema);
      tools = [
        { function_declarations: funcs.map((schema) => schema.function) },
      ];
    }
  }
  if (req.tool_choice) {
    const allowed_function_names =
      req.tool_choice?.type === "function"
        ? [req.tool_choice?.function?.name]
        : undefined;
    if (allowed_function_names || typeof req.tool_choice === "string") {
      tool_config = {
        function_calling_config: {
          mode: allowed_function_names ? "ANY" : req.tool_choice.toUpperCase(),
          allowed_function_names,
        },
      };
    }
  }
  return { tools, tool_config };
}

async function transformRequest(req) {
  return {
    ...(await transformMessages(req.messages)),
    safetySettings,
    generationConfig: transformConfig(req),
    ...transformTools(req),
  };
}

const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
async function handleEmbeddings(req, apiKey) {
  if (typeof req.model !== "string")
    throw new HttpError("model is not specified", 400);
  let model;
  if (req.model.startsWith("models/")) {
    model = req.model;
  } else {
    if (!req.model.startsWith("gemini-")) req.model = DEFAULT_EMBEDDINGS_MODEL;
    model = "models/" + req.model;
  }
  if (!Array.isArray(req.input)) req.input = [req.input];
  const response = await fetch(
    `${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`,
    {
      method: "POST",
      headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
      body: JSON.stringify({
        requests: req.input.map((text) => ({
          model,
          content: { parts: { text } },
          outputDimensionality: req.dimensions,
        })),
      }),
    }
  );
  let { body } = response;
  if (response.ok) {
    const { embeddings } = JSON.parse(await response.text());
    body = JSON.stringify(
      {
        object: "list",
        data: embeddings.map(({ values }, index) => ({
          object: "embedding",
          index,
          embedding: values,
        })),
        model: req.model,
      },
      null,
      "  "
    );
  }
  return new Response(body, fixCors(response));
}

async function handleModels(apiKey) {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  let { body } = response;
  if (response.ok) {
    const { models } = JSON.parse(await response.text());
    body = JSON.stringify(
      {
        object: "list",
        data: models.map(({ name }) => ({
          id: name.replace("models/", ""),
          object: "model",
          created: 0,
          owned_by: "",
        })),
      },
      null,
      "  "
    );
  }
  return new Response(body, fixCors(response));
}

const DEFAULT_MODEL = "gemini-2.5-flash";
async function handleCompletions(req, apiKey) {
  let model = DEFAULT_MODEL;
  if (typeof req.model === "string") {
    if (req.model.startsWith("models/")) model = req.model.substring(7);
    else if (
      req.model.startsWith("gemini-") ||
      req.model.startsWith("gemma-") ||
      req.model.startsWith("learnlm-")
    )
      model = req.model;
  }
  let body = await transformRequest(req);
  const extra = req.extra_body?.google;
  if (extra) {
    if (extra.safety_settings) body.safetySettings = extra.safety_settings;
    if (extra.cached_content) body.cachedContent = extra.cached_content;
    if (extra.thinking_config)
      body.generationConfig.thinkingConfig = extra.thinking_config;
  }
  if (
    model.endsWith(":search") ||
    req.model?.endsWith("-search-preview") ||
    req.tools?.some((t) => t.function?.name === "googleSearch")
  ) {
    body.tools = body.tools || [];
    body.tools.push({ googleSearch: {} });
  }
  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) url += "?alt=sse";
  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });

  body = response.body;
  if (response.ok) {
    const shared = {};
    const id = "chatcmpl-" + generateId();
    if (req.stream) {
      body = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(
          new TransformStream({
            transform: parseStream,
            flush: parseStreamFlush,
            buffer: "",
            shared,
          })
        )
        .pipeThrough(
          new TransformStream({
            transform: toOpenAiStream,
            flush: toOpenAiStreamFlush,
            streamIncludeUsage: req.stream_options?.include_usage,
            model,
            id,
            last: [],
            shared,
          })
        )
        .pipeThrough(new TextEncoderStream());
    } else {
      body = await response.text();
      try {
        body = JSON.parse(body);
        if (!body.candidates) throw new Error("Invalid completion object");
      } catch (err) {
        console.error("Error parsing response:", err);
        return new Response(body, fixCors(response));
      }
      body = processCompletionsResponse(body, model, id);
    }
  }
  return new Response(body, fixCors(response));
}

function processCompletionsResponse(data, model, id) {
  const obj = {
    id,
    choices: data.candidates.map((c) => transformCandidates("message", c)),
    created: Math.floor(Date.now() / 1000),
    model: data.modelVersion ?? model,
    object: "chat.completion",
    usage: data.usageMetadata && transformUsage(data.usageMetadata),
  };
  if (obj.choices.length === 0) {
    checkPromptBlock(obj.choices, data.promptFeedback, "message");
  }
  return JSON.stringify(obj);
}

function checkPromptBlock(choices, promptFeedback, key) {
  if (choices.length) return;
  if (promptFeedback?.blockReason) {
    if (promptFeedback.blockReason === "SAFETY") {
      promptFeedback.safetyRatings
        ?.filter((r) => r.blocked)
        .forEach((r) => console.log(r));
    }
    choices.push({ index: 0, [key]: null, finish_reason: "content_filter" });
  }
  return true;
}

const responseLineRE = /^data: (.*)(?:\n\n|\r\r|\r\n\r\n)/;
function parseStream(chunk, controller) {
  this.buffer += chunk;
  do {
    const match = this.buffer.match(responseLineRE);
    if (!match) break;
    controller.enqueue(match[1]);
    this.buffer = this.buffer.substring(match[0].length);
  } while (true);
}
function parseStreamFlush(controller) {
  if (this.buffer) {
    console.error("Invalid data:", this.buffer);
    controller.enqueue(this.buffer);
    this.shared.is_buffers_rest = true;
  }
}

const delimiter = "\n\n";
function sseline(obj) {
  obj.created = Math.floor(Date.now() / 1000);
  return "data: " + JSON.stringify(obj) + delimiter;
}
function toOpenAiStream(line, controller) {
  let data;
  try {
    data = JSON.parse(line);
    if (!data.candidates) throw new Error("Invalid completion chunk object");
  } catch (err) {
    console.error("Error parsing response:", err);
    if (!this.shared.is_buffers_rest) {
      line = +delimiter;
    }
    controller.enqueue(line);
    return;
  }
  const obj = {
    id: this.id,
    choices: data.candidates.map((c) => transformCandidates("delta", c)),
    model: data.modelVersion ?? this.model,
    object: "chat.completion.chunk",
    usage: data.usageMetadata && this.streamIncludeUsage ? null : undefined,
  };
  if (checkPromptBlock(obj.choices, data.promptFeedback, "delta")) {
    controller.enqueue(sseline(obj));
    return;
  }
  const cand = obj.choices[0];
  cand.index = cand.index || 0;
  const finish_reason = cand.finish_reason;
  cand.finish_reason = null;
  if (!this.last[cand.index]) {
    controller.enqueue(
      sseline({
        ...obj,
        choices: [
          {
            ...cand,
            tool_calls: undefined,
            delta: { role: "assistant", content: "" },
          },
        ],
      })
    );
  }
  delete cand.delta.role;
  if ("content" in cand.delta) {
    controller.enqueue(sseline(obj));
  }
  cand.finish_reason = finish_reason;
  if (data.usageMetadata && this.streamIncludeUsage) {
    obj.usage = transformUsage(data.usageMetadata);
  }
  cand.delta = {};
  this.last[cand.index] = obj;
}
function toOpenAiStreamFlush(controller) {
  if (this.last.length > 0) {
    for (const obj of this.last) {
      controller.enqueue(sseline(obj));
    }
    controller.enqueue("data: [DONE]" + delimiter);
  }
}

async function openaiAdapterFetch(request) {
  const errHandler = (err) => {
    console.error(err);
    return new Response(
      err.message || "Internal Error",
      fixCors({ status: err.status ?? 500 })
    );
  };
  try {
    const auth = request.headers.get("Authorization");
    let apiKey = auth?.split(" ")[1];
    if (apiKey && apiKey.includes(",")) {
      const picked = selectOneFromCommaList(apiKey);
      console.log(`OpenAI Selected API Key: ${maskKey(picked)}`);
      apiKey = picked;
    }
    const { pathname } = new URL(request.url);
    switch (true) {
      case pathname.endsWith("/chat/completions"): {
        if (request.method !== "POST")
          throw new HttpError(
            "The specified HTTP method is not allowed for the requested resource",
            400
          );
        return handleCompletions(await request.json(), apiKey).catch(
          errHandler
        );
      }
      case pathname.endsWith("/embeddings"): {
        if (request.method !== "POST")
          throw new HttpError(
            "The specified HTTP method is not allowed for the requested resource",
            400
          );
        return handleEmbeddings(await request.json(), apiKey).catch(errHandler);
      }
      case pathname.endsWith("/models"): {
        if (request.method !== "GET")
          throw new HttpError(
            "The specified HTTP method is not allowed for the requested resource",
            400
          );
        return handleModels(apiKey).catch(errHandler);
      }
      default:
        throw new HttpError("404 Not Found", 404);
    }
  } catch (err) {
    return errHandler(err);
  }
}
