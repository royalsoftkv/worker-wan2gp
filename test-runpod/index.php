<?php
/**
 * RunPod endpoint tester. Submit via /run, poll /status/{id} until done.
 * Edit config below.
 */
$RUNPOD_CONFIG = [
    'endpoint'           => 'https://api.runpod.ai/v2/YOUR_ENDPOINT_ID',  // base URL, no /run
    'api_key'            => 'YOUR_RUNPOD_API_KEY',
    'poll_interval_sec'  => 2,
    'poll_timeout_sec'   => 600,
];

$config = $RUNPOD_CONFIG;
$base = rtrim(trim($config['endpoint'] ?? ''), '/');
$apiKey = trim($config['api_key'] ?? '');
$headers = ['Content-Type: application/json'];
if ($apiKey) $headers[] = 'Authorization: Bearer ' . $apiKey;

// Submit job (async): POST payload → return job id only
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['payload'])) {
    header('Content-Type: application/json');
    $payload = json_decode($_POST['payload'], true);
    if (!$base || !$payload) {
        echo json_encode(['error' => 'Set endpoint and api_key in index.php ($RUNPOD_CONFIG) and provide payload']);
        exit;
    }
    $runUrl = $base . '/run';
    $ch = curl_init($runUrl);
    curl_setopt_array($ch, [
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => json_encode($payload),
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER => $headers,
        CURLOPT_TIMEOUT => 60,
    ]);
    $response = curl_exec($ch);
    $err = curl_error($ch);
    curl_close($ch);
    if ($err) {
        echo json_encode(['error' => 'Run request failed', 'details' => $err]);
        exit;
    }
    $data = json_decode($response, true) ?: [];
    $jobId = $data['id'] ?? null;
    if (!$jobId) {
        echo json_encode(['error' => 'No job id in response', 'response' => $data, 'raw' => $response]);
        exit;
    }
    echo json_encode(['id' => $jobId]);
    exit;
}

// Poll status once: GET ?id=xxx → return RunPod status response
if ($_SERVER['REQUEST_METHOD'] === 'GET' && isset($_GET['id']) && $base) {
    header('Content-Type: application/json');
    $jobId = trim($_GET['id']);
    if (!$jobId) {
        echo json_encode(['error' => 'Missing id']);
        exit;
    }
    $statusUrl = $base . '/status/' . $jobId;
    $ch = curl_init($statusUrl);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_HTTPHEADER => $apiKey ? ['Authorization: Bearer ' . $apiKey] : [],
        CURLOPT_TIMEOUT => 30,
    ]);
    $response = curl_exec($ch);
    curl_close($ch);
    if ($response === false) {
        echo json_encode(['error' => 'Status request failed']);
        exit;
    }
    echo $response;
    exit;
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RunPod endpoint tester</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" />
    <script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
</head>
<body class="bg-slate-100 min-h-screen p-6">
    <div id="app" class="max-w-2xl mx-auto">
        <div class="bg-white rounded-xl shadow-lg p-6">
            <h1 class="text-xl font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <i class="fas fa-rocket text-indigo-500"></i>
                RunPod endpoint tester
            </h1>

            <p class="text-sm text-slate-500 mb-4">Endpoint and API key are set in <code class="bg-slate-100 px-1 rounded">index.php</code> (config at top).</p>
            <div class="space-y-4">
                <div>
                    <label class="block text-sm font-medium text-slate-600 mb-1">Mode</label>
                    <div class="flex gap-4">
                        <label class="flex items-center gap-2 cursor-pointer">
                            <input type="radio" v-model="mode" value="image" class="text-indigo-600">
                            <span>Image</span>
                        </label>
                        <label class="flex items-center gap-2 cursor-pointer">
                            <input type="radio" v-model="mode" value="video">
                            <span>Video</span>
                        </label>
                    </div>
                </div>
                <div v-if="mode === 'image'">
                    <label class="block text-sm font-medium text-slate-600 mb-1">Prompt</label>
                    <textarea v-model="prompt" rows="2" placeholder="A photo of..."
                        class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500"></textarea>
                </div>
                <div v-if="mode === 'video'">
                    <label class="block text-sm font-medium text-slate-600 mb-1">Prompt</label>
                    <textarea v-model="prompt" rows="2" placeholder="Video prompt..."
                        class="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500"></textarea>
                    <label class="block text-sm font-medium text-slate-600 mt-2 mb-1">Source image (upload)</label>
                    <div class="flex items-center gap-3">
                        <input type="file" accept="image/*" @change="onVideoImageUpload" ref="videoImageInput"
                            class="block w-full text-sm text-slate-500 file:mr-3 file:py-2 file:px-3 file:rounded-lg file:border-0 file:bg-indigo-50 file:text-indigo-700 file:font-medium">
                        <img v-if="videoImagePreview" :src="videoImagePreview" alt="Upload" class="h-16 w-16 object-cover rounded border border-slate-200">
                    </div>
                    <p v-if="videoImageName" class="text-xs text-slate-500 mt-1">{{ videoImageName }} (used as input for video)</p>
                </div>
                <div>
                    <label class="block text-sm font-medium text-slate-600 mb-1">Payload (JSON)</label>
                    <textarea v-model="payloadStr" rows="8" placeholder='{"input": {...}}'
                        class="w-full rounded-lg border border-slate-300 px-3 py-2 font-mono text-sm focus:ring-2 focus:ring-indigo-500"></textarea>
                    <p v-if="payloadError" class="text-xs text-red-600 mt-1">{{ payloadError }}</p>
                </div>
                <button @click="run" :disabled="loading"
                    class="w-full rounded-lg bg-indigo-600 text-white py-2.5 font-medium hover:bg-indigo-700 disabled:opacity-50 flex items-center justify-center gap-2">
                    <i v-if="loading" class="fas fa-spinner fa-spin"></i>
                    <i v-else class="fas fa-play"></i>
                    {{ loading ? (jobId ? 'Polling... ' + jobId : 'Submitting...') : 'Run' }}
                </button>
            </div>

            <div v-if="error" class="mt-4 p-3 rounded-lg bg-red-50 text-red-800 text-sm flex items-start gap-2">
                <i class="fas fa-exclamation-circle mt-0.5"></i>
                <span>{{ error }}</span>
            </div>
            <div v-if="result" class="mt-4 space-y-3">
                <p class="text-sm font-medium text-slate-600">Result</p>
                <img v-if="resultImage" :src="resultImage" alt="Output" class="rounded-lg border border-slate-200 max-w-full max-h-96 object-contain bg-slate-50">
                <video v-else-if="resultVideo" :src="resultVideo" controls class="rounded-lg border border-slate-200 max-w-full max-h-96 bg-slate-900"></video>
                <pre v-else class="p-3 rounded-lg bg-slate-800 text-slate-100 text-xs overflow-auto max-h-64">{{ resultText }}</pre>
            </div>
        </div>
    </div>

    <script>
        const { createApp } = Vue;
        createApp({
            data() {
                return {
                    mode: 'image',
                    prompt: 'a photo of a cat on a windowsill',
                    payloadStr: '',
                    videoImageBase64: null,
                    videoImagePreview: null,
                    videoImageName: null,
                    loading: false,
                    jobId: null,
                    error: null,
                    result: null,
                    resultImage: null,
                    resultVideo: null,
                    resultText: null,
                };
            },
            watch: {
                mode() {
                    this.videoImageBase64 = null;
                    this.videoImagePreview = null;
                    this.videoImageName = null;
                    this.setDefaultPayload();
                },
                prompt() {
                    this.updatePayloadFromPrompt();
                },
            },
            mounted() {
                this.setDefaultPayload();
            },
            computed: {
                payloadError() {
                    if (!this.payloadStr.trim()) return null;
                    try {
                        JSON.parse(this.payloadStr);
                        return null;
                    } catch (e) {
                        return 'Invalid JSON: ' + e.message;
                    }
                },
            },
            methods: {
                setDefaultPayload() {
                    if (this.mode === 'image') {
                        this.payloadStr = JSON.stringify({
                            input: {
                                prompt: this.prompt || 'a photo of a cat',
                                negative_prompt: '',
                                width: 512,
                                height: 512,
                                num_inference_steps: 11,
                                guidance_scale: 7,
                                sampler: 'DPM++ 2M Karras',
                            },
                        }, null, 2);
                    } else {
                        this.payloadStr = JSON.stringify({
                            input: {
                                prompt: this.prompt || 'video prompt',
                                negative_prompt: '',
                                image: this.videoImageBase64 || '',
                                resolution: '512x512',
                                video_length: 81,
                                num_inference_steps: 10,
                                guidance_scale: 5,
                                flow_shift: 5,
                            },
                        }, null, 2);
                    }
                },
                onVideoImageUpload(e) {
                    const file = e.target.files && e.target.files[0];
                    if (!file || !file.type.startsWith('image/')) return;
                    this.videoImageName = file.name;
                    const reader = new FileReader();
                    reader.onload = () => {
                        const dataUrl = reader.result;
                        this.videoImagePreview = dataUrl;
                        this.videoImageBase64 = dataUrl.replace(/^data:image\/\w+;base64,/, '');
                        this.updatePayloadWithVideoImage();
                    };
                    reader.readAsDataURL(file);
                },
                updatePayloadWithVideoImage() {
                    if (!this.videoImageBase64) return;
                    try {
                        const p = JSON.parse(this.payloadStr);
                        if (p.input) p.input.image = this.videoImageBase64;
                        this.payloadStr = JSON.stringify(p, null, 2);
                    } catch (_) {}
                },
                updatePayloadFromPrompt() {
                    try {
                        const p = JSON.parse(this.payloadStr);
                        if (p.input) p.input.prompt = this.prompt;
                        this.payloadStr = JSON.stringify(p, null, 2);
                    } catch (_) {}
                },
                async run() {
                    this.error = null;
                    this.result = null;
                    this.resultImage = null;
                    this.resultVideo = null;
                    this.resultText = null;
                    this.jobId = null;
                    let payload;
                    try {
                        payload = JSON.parse(this.payloadStr);
                    } catch (e) {
                        this.error = 'Invalid JSON: ' + e.message;
                        return;
                    }
                    if (this.mode === 'video' && (!payload.input || !payload.input.image)) {
                        this.error = 'Upload a source image for video';
                        return;
                    }
                    this.loading = true;
                    const form = new FormData();
                    form.append('payload', JSON.stringify(payload));
                    const pollIntervalMs = 2000;
                    const pollTimeoutMs = 600000; // 10 min
                    try {
                        const r = await fetch('index.php', { method: 'POST', body: form });
                        const submitData = await r.json().catch(() => ({}));
                        if (!r.ok) {
                            this.error = submitData.error || submitData.details || 'Request failed';
                            this.resultText = JSON.stringify(submitData, null, 2);
                            return;
                        }
                        const jobId = submitData.id;
                        if (!jobId) {
                            this.error = 'No job id returned';
                            return;
                        }
                        this.jobId = jobId;
                        const deadline = Date.now() + pollTimeoutMs;
                        while (Date.now() < deadline) {
                            const sr = await fetch('index.php?id=' + encodeURIComponent(jobId));
                            const data = await sr.json().catch(() => ({}));
                            const status = data.status || '';
                            if (status === 'COMPLETED') {
                                this.result = data;
                                const out = data.output != null ? data.output : data;
                                if (out && out.image_base64) {
                                    this.resultImage = 'data:image/png;base64,' + out.image_base64;
                                } else if (out && out.video_base64) {
                                    this.resultVideo = 'data:video/mp4;base64,' + out.video_base64;
                                } else if (out && (out.error || out.details)) {
                                    this.error = out.error || out.details;
                                    this.resultText = JSON.stringify(out, null, 2);
                                } else {
                                    this.resultText = JSON.stringify(data, null, 2);
                                }
                                this.jobId = null;
                                return;
                            }
                            if (status === 'FAILED') {
                                this.error = 'Job failed';
                                this.resultText = JSON.stringify(data, null, 2);
                                this.jobId = null;
                                return;
                            }
                            await new Promise(r => setTimeout(r, pollIntervalMs));
                        }
                        this.error = 'Job timeout';
                        this.jobId = null;
                    } catch (e) {
                        this.error = e.message || 'Network error';
                        this.jobId = null;
                    } finally {
                        this.loading = false;
                    }
                },
            },
        }).mount('#app');
    </script>
</body>
</html>
