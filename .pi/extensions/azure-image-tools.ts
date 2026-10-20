import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

const AZURE_IMAGE_PROVIDER = "azure-images";
const AZURE_IMAGE_API = "azure-openai-images";
const DEFAULT_MODEL_ID = "gpt-image-2.5-flare";
const FALLBACK_MODEL_ID = "gpt-image-2.5-sunburst";
const IMAGE_TOOL_NAMES = new Set(["generate_image", "codemode"]);

type ImageBlock = { type: "image"; data: string; mimeType: string };
type TextBlock = { type: "text"; text: string };
type ToolContentBlock = ImageBlock | TextBlock;

type GenerateImageParams = {
  prompt: string;
  model?: string;
  size?: "auto" | "1024x1024" | "1536x1024" | "1024x1536";
  quality?: "auto" | "low" | "medium" | "high";
  background?: "auto" | "transparent" | "opaque";
  format?: "png" | "jpeg" | "webp";
  count?: number;
};

type GenerateImageResult = {
  api: string;
  provider: string;
  model: string;
  output: ToolContentBlock[];
  stopReason: "stop" | "error" | "aborted";
  errorMessage?: string;
  usage?: unknown;
  responseId?: string;
  timestamp: number;
};

function trimTrailingSlash(value: string) {
  return value.replace(/\/+$/, "");
}

function ensureAzureOpenAiV1BaseUrl(value: string) {
  let baseUrl = trimTrailingSlash(value.trim());
  baseUrl = baseUrl.replace(/\/images\/generations$/i, "");

  if (/\/openai\/v\d+$/i.test(baseUrl)) return baseUrl;
  if (/\/openai$/i.test(baseUrl)) return `${baseUrl}/v1`;
  return `${baseUrl}/openai/v1`;
}

function resolveAzureBaseUrl(model: { baseUrl?: string }, options?: { env?: Record<string, string | undefined> }) {
  const env = options?.env || process.env;
  const configuredBaseUrl = String(model.baseUrl || env.AZURE_OPENAI_BASE_URL || "").trim();
  if (configuredBaseUrl) return ensureAzureOpenAiV1BaseUrl(configuredBaseUrl);

  const resourceName = String(env.AZURE_OPENAI_RESOURCE_NAME || "").trim();
  if (resourceName) {
    return `https://${resourceName}.services.ai.azure.com/openai/v1`;
  }

  throw new Error(
    "Azure OpenAI base URL is not configured. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME.",
  );
}

const AUTH_JSON_KEY_COMMAND = `!python -c "import json, os; print(json.load(open(os.path.expanduser('~/.pi/agent/auth.json')))['azure']['key'])"`;

function getApiKey(options?: { apiKey?: string; env?: Record<string, string | undefined> }) {
  const apiKey = String(options?.apiKey || options?.env?.AZURE_OPENAI_API_KEY || process.env.AZURE_OPENAI_API_KEY || "").trim();
  if (!apiKey) {
    throw new Error("Azure OpenAI API key is not available for image generation.");
  }
  return apiKey;
}

function toTextContent(value: unknown) {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function normalizeMimeType(value: unknown, fallback = "image/png") {
  const mimeType = String(value || "").trim().toLowerCase();
  if (!mimeType) return fallback;
  if (mimeType === "jpg") return "image/jpeg";
  if (mimeType === "jpeg" || mimeType === "png" || mimeType === "webp") return `image/${mimeType}`;
  if (mimeType.startsWith("image/")) return mimeType;
  return fallback;
}

function parseDataUrl(dataUrl: string) {
  const match = /^data:([^;,]+);base64,(.+)$/i.exec(dataUrl.trim());
  if (!match) return null;
  return {
    mimeType: normalizeMimeType(match[1]),
    data: match[2],
  };
}

async function responseToErrorMessage(response: Response) {
  let payload: any = null;
  let rawText = "";

  try {
    rawText = await response.text();
    payload = rawText ? JSON.parse(rawText) : null;
  } catch {
    payload = null;
  }

  const providerMessage =
    payload?.error?.message ||
    payload?.message ||
    rawText ||
    `${response.status} ${response.statusText}`;

  return `Azure image generation failed (${response.status}): ${String(providerMessage).trim()}`;
}

async function extractOutputBlocks(payload: any, outputFormat = "png") {
  const blocks: ToolContentBlock[] = [];
  const images: ImageBlock[] = [];
  const fallbackMimeType = normalizeMimeType(outputFormat, "image/png");
  const items = Array.isArray(payload?.data)
    ? payload.data
    : Array.isArray(payload?.output)
      ? payload.output
      : [];

  for (const item of items) {
    if (!item || typeof item !== "object") continue;

    const base64Data =
      typeof item.b64_json === "string" ? item.b64_json
      : typeof item.base64 === "string" ? item.base64
      : typeof item.image_base64 === "string" ? item.image_base64
      : typeof item.result === "string" ? item.result
      : null;

    if (base64Data) {
      images.push({
        type: "image",
        data: base64Data,
        mimeType: normalizeMimeType(item.mime_type || item.mimeType || item.output_format, fallbackMimeType),
      });
    }

    const dataUrl =
      typeof item.url === "string" && item.url.startsWith("data:") ? item.url
      : typeof item.image_url === "string" && item.image_url.startsWith("data:") ? item.image_url
      : null;

    if (dataUrl) {
      const parsed = parseDataUrl(dataUrl);
      if (parsed) images.push({ type: "image", ...parsed });
    }

    if (typeof item.revised_prompt === "string" && item.revised_prompt.trim()) {
      blocks.push({ type: "text", text: `Revised prompt: ${item.revised_prompt.trim()}` });
    }
  }

  if (!images.length && typeof payload?.revised_prompt === "string" && payload.revised_prompt.trim()) {
    blocks.push({ type: "text", text: `Revised prompt: ${payload.revised_prompt.trim()}` });
  }

  return [...blocks, ...images];
}

async function generateAzureOpenAiImages(
  model: { id: string; api: string; provider: string; baseUrl?: string },
  context: { input: Array<{ type: string; text?: string }> },
  options?: {
    apiKey?: string;
    env?: Record<string, string | undefined>;
    fetch?: typeof fetch;
    headers?: Record<string, string | null | undefined>;
    metadata?: Record<string, unknown>;
    signal?: AbortSignal;
  },
): Promise<GenerateImageResult> {
  const startedAt = Date.now();

  try {
    const prompt = context.input
      .filter((block) => block?.type === "text" && typeof block.text === "string")
      .map((block) => block.text?.trim() || "")
      .filter(Boolean)
      .join("\n\n");

    if (!prompt) {
      return {
        api: model.api,
        provider: model.provider,
        model: model.id,
        output: [],
        stopReason: "error",
        errorMessage: "Image generation requires a text prompt.",
        timestamp: startedAt,
      };
    }

    const env = options?.env || process.env;
    const fetchImpl = options?.fetch || fetch;
    const apiKey = getApiKey(options);
    const baseUrl = resolveAzureBaseUrl(model, { env });
    const endpoint = `${baseUrl}/images/generations`;

    const metadata = options?.metadata || {};
    const outputFormat = typeof metadata.output_format === "string" ? metadata.output_format : "png";
    const payload: Record<string, unknown> = {
      model: model.id,
      prompt,
      output_format: outputFormat,
    };

    for (const key of ["size", "quality", "background", "n", "user"]) {
      const value = metadata[key];
      if (value !== undefined && value !== null && value !== "") payload[key] = value;
    }

    const headers: Record<string, string> = {
      "content-type": "application/json",
      "api-key": apiKey,
    };

    for (const [key, value] of Object.entries(options?.headers || {})) {
      if (typeof value === "string") headers[key] = value;
    }

    const response = await fetchImpl(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
      signal: options?.signal,
    });

    if (!response.ok) {
      return {
        api: model.api,
        provider: model.provider,
        model: model.id,
        output: [],
        stopReason: "error",
        errorMessage: await responseToErrorMessage(response),
        timestamp: startedAt,
      };
    }

    const json = await response.json();
    const output = await extractOutputBlocks(json, outputFormat);
    const hasImage = output.some((block) => block.type === "image");

    if (!hasImage) {
      return {
        api: model.api,
        provider: model.provider,
        model: model.id,
        output: [],
        stopReason: "error",
        errorMessage: "Azure image generation returned no images.",
        timestamp: startedAt,
      };
    }

    return {
      api: model.api,
      provider: model.provider,
      model: model.id,
      output,
      responseId: typeof json?.id === "string" ? json.id : undefined,
      usage: json?.usage,
      stopReason: "stop",
      timestamp: startedAt,
    };
  } catch (error: any) {
    const aborted = error?.name === "AbortError";
    return {
      api: model.api,
      provider: model.provider,
      model: model.id,
      output: [],
      stopReason: aborted ? "aborted" : "error",
      errorMessage: error instanceof Error ? error.message : String(error),
      timestamp: startedAt,
    };
  }
}

function normalizeAssistantContent(content: unknown) {
  if (typeof content === "string") {
    return content.trim() ? [{ type: "text", text: content }] : [];
  }
  if (Array.isArray(content)) {
    return content.filter((block) => block && typeof block === "object");
  }
  return [];
}

function cloneImageBlock(block: any): ImageBlock {
  return {
    type: "image",
    data: String(block.data || ""),
    mimeType: normalizeMimeType(block.mimeType),
  };
}

const generateImageTool = defineTool({
  name: "generate_image",
  label: "Generate Image",
  description:
    "Generate an image with Azure image models. Use this when the user asks you to create or render an image, illustration, or artwork.",
  parameters: Type.Object({
    prompt: Type.String({ minLength: 1, description: "The image prompt to render" }),
    model: Type.Optional(
      Type.Union([
        Type.Literal(DEFAULT_MODEL_ID),
        Type.Literal(FALLBACK_MODEL_ID),
      ], { description: `Optional image model. Defaults to ${DEFAULT_MODEL_ID}.` }),
    ),
    size: Type.Optional(
      Type.Union([
        Type.Literal("auto"),
        Type.Literal("1024x1024"),
        Type.Literal("1536x1024"),
        Type.Literal("1024x1536"),
      ], { description: "Optional output size" }),
    ),
    quality: Type.Optional(
      Type.Union([
        Type.Literal("auto"),
        Type.Literal("low"),
        Type.Literal("medium"),
        Type.Literal("high"),
      ], { description: "Optional render quality" }),
    ),
    background: Type.Optional(
      Type.Union([
        Type.Literal("auto"),
        Type.Literal("transparent"),
        Type.Literal("opaque"),
      ], { description: "Optional background mode" }),
    ),
    format: Type.Optional(
      Type.Union([
        Type.Literal("png"),
        Type.Literal("jpeg"),
        Type.Literal("webp"),
      ], { description: "Output image format" }),
    ),
    count: Type.Optional(Type.Integer({ minimum: 1, maximum: 4, description: "Number of images to request" })),
  }),
  async execute(_toolCallId, params: GenerateImageParams, signal, _onUpdate, ctx) {
    const availableModels = await ctx.modelRegistry.getAvailableOfType("image", AZURE_IMAGE_PROVIDER, { signal });
    if (!availableModels.length) {
      return {
        content: [{ type: "text", text: "No Azure image models are currently available." }],
        details: { provider: AZURE_IMAGE_PROVIDER, available: false },
        isError: true,
      };
    }

    const requestedModelId = params.model || DEFAULT_MODEL_ID;
    const selectedModel =
      availableModels.find((model: any) => model.id === requestedModelId) ||
      availableModels.find((model: any) => model.id === DEFAULT_MODEL_ID) ||
      availableModels.find((model: any) => model.id === FALLBACK_MODEL_ID) ||
      availableModels[0];

    const result = await ctx.modelRegistry.generateImages(
      selectedModel,
      {
        input: [{ type: "text", text: params.prompt.trim() }],
      },
      {
        signal,
        metadata: {
          size: params.size,
          quality: params.quality,
          background: params.background,
          output_format: params.format || "png",
          n: params.count,
        },
      },
    );

    if (result.stopReason !== "stop") {
      return {
        content: [{ type: "text", text: result.errorMessage || "Image generation failed." }],
        details: {
          provider: result.provider,
          model: result.model,
          stopReason: result.stopReason,
          responseId: result.responseId,
        },
        usage: result.usage,
        isError: true,
      };
    }

    const images = result.output.filter((block: any) => block?.type === "image");
    const textBlocks = result.output.filter((block: any) => block?.type === "text");
    const summary = `Generated ${images.length} image${images.length === 1 ? "" : "s"} with ${result.model}.`;

    return {
      content: [
        { type: "text", text: summary },
        ...textBlocks.map((block: any) => ({ type: "text", text: toTextContent(block.text || "") })),
        ...images.map(cloneImageBlock),
      ],
      details: {
        provider: result.provider,
        model: result.model,
        responseId: result.responseId,
        imageCount: images.length,
      },
      usage: result.usage,
    };
  },
});

export default function registerAzureImageTools(pi: ExtensionAPI) {
  let pendingAssistantImages: ImageBlock[] = [];
  const providerBaseUrl = resolveAzureBaseUrl({ baseUrl: "" }, { env: process.env });

  pi.registerProvider(AZURE_IMAGE_PROVIDER, {
    name: "Azure Images",
    baseUrl: providerBaseUrl,
    apiKey: AUTH_JSON_KEY_COMMAND,
    models: [
      {
        type: "image",
        id: DEFAULT_MODEL_ID,
        name: "GPT Image 2.5 Flare",
        api: AZURE_IMAGE_API,
        baseUrl: providerBaseUrl,
        input: ["text", "image"],
        output: ["image"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      },
      {
        type: "image",
        id: FALLBACK_MODEL_ID,
        name: "GPT Image 2.5 Sunburst",
        api: AZURE_IMAGE_API,
        baseUrl: providerBaseUrl,
        input: ["text", "image"],
        output: ["image"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      },
    ],
    images: {
      [AZURE_IMAGE_API]: {
        generateImages: generateAzureOpenAiImages,
      },
    },
  });

  pi.registerTool(generateImageTool);

  pi.on("turn_start", () => {
    pendingAssistantImages = [];
  });

  pi.on("tool_result", (event) => {
    if (event.isError || !IMAGE_TOOL_NAMES.has(event.toolName)) return;
    const images = event.content.filter((block) => block?.type === "image");
    if (!images.length) return;
    pendingAssistantImages.push(...images.map(cloneImageBlock));
  });

  pi.on("message_end", (event) => {
    if (event.message.role !== "assistant" || pendingAssistantImages.length === 0) return;

    const content = normalizeAssistantContent(event.message.content);
    const message = {
      ...event.message,
      content: [...content, ...pendingAssistantImages.map(cloneImageBlock)],
    };

    pendingAssistantImages = [];
    return { message };
  });
}
