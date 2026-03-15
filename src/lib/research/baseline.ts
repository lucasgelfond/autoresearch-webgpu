/**
 * The baseline training code — equivalent to the initial train.py.
 * This is what runs on the first experiment before Claude starts iterating.
 */
export const BASELINE_CODE = `// GPT model — byte-level transformer
const nLayer = 3, nEmbd = 96, nHead = 4, mlpRatio = 4;
const headDim = nEmbd / nHead, mlpHidden = nEmbd * mlpRatio;
const lr = 1e-3, weightDecay = 0.1, warmupRatio = 0.1, cooldownRatio = 0.3;
const batchSize = 8, seqLen = 128;

// --- init params ---
const key = random.key(42);
const numKeys = 3 + nLayer * 8;
const keys = random.split(key, numKeys);
let ki = 0;
const grabKey = () => { ki++; return ki < numKeys ? keys.ref.slice(ki - 1) : keys.slice(ki - 1); };

const params = {};
params['embed'] = random.normal(grabKey(), [VOCAB_SIZE, nEmbd]).mul(1.0);
for (let i = 0; i < nLayer; i++) {
  const s = Math.sqrt(3) * Math.pow(nEmbd, -0.5);
  const p = 'layer' + i;
  params[p + '.attn.wq'] = random.uniform(grabKey(), [nEmbd, nHead * headDim], { minval: -s, maxval: s });
  params[p + '.attn.wk'] = random.uniform(grabKey(), [nEmbd, nHead * headDim], { minval: -s, maxval: s });
  params[p + '.attn.wv'] = random.uniform(grabKey(), [nEmbd, nHead * headDim], { minval: -s, maxval: s });
  params[p + '.attn.wout'] = np.zeros([nEmbd, nEmbd]); grabKey();
  params[p + '.norm1'] = np.ones([nEmbd]); grabKey();
  params[p + '.norm2'] = np.ones([nEmbd]); grabKey();
  params[p + '.mlp.up'] = random.uniform(grabKey(), [nEmbd, mlpHidden], { minval: -s, maxval: s });
  params[p + '.mlp.down'] = np.zeros([mlpHidden, nEmbd]); grabKey();
}
params['final_norm'] = np.ones([nEmbd]);
params['unembed'] = random.normal(grabKey(), [nEmbd, VOCAB_SIZE]).mul(0.001);
await blockUntilReady(params);

// --- helpers ---
function rmsNorm(x, w) { return nn.standardize(x, -1, { epsilon: 1e-6 }).mul(w); }

function ropeFreqs(sl, hd) {
  const half = hd / 2;
  const freqs = np.power(10000, np.negative(np.arange(0, half, 1, { dtype: np.float32 }).mul(2 / hd)));
  const pos = np.arange(0, sl, 1, { dtype: np.float32 });
  const angles = np.outer(pos, freqs);
  return [np.cos(angles.ref), np.sin(angles)];
}

function applyRoPE(x, cos, sin) {
  const half = x.shape[3] / 2;
  const x1 = x.ref.slice([], [], [], [0, half]);
  const x2 = x.slice([], [], [], [half]);
  const c = cos.reshape([1, -1, 1, half]);
  const s = sin.reshape([1, -1, 1, half]);
  return np.concatenate([x1.ref.mul(c.ref).sub(x2.ref.mul(s.ref)), x1.mul(c).add(x2.mul(s))], -1);
}

// --- forward pass ---
function forward(p, inputIds) {
  const [_b, sl] = inputIds.shape;
  const ids = nn.oneHot(inputIds.reshape([-1]), VOCAB_SIZE);
  let x = np.dot(ids, p['embed'].ref).reshape([-1, sl, nEmbd]);
  const [rc, rs] = ropeFreqs(sl, headDim);
  for (let i = 0; i < nLayer; i++) {
    const pfx = 'layer' + i, last = i === nLayer - 1;
    const n1 = rmsNorm(x.ref, p[pfx + '.norm1'].ref);
    let q = np.dot(n1.ref, p[pfx + '.attn.wq'].ref).reshape([-1, sl, nHead, headDim]);
    let k = np.dot(n1.ref, p[pfx + '.attn.wk'].ref).reshape([-1, sl, nHead, headDim]);
    const v = np.dot(n1, p[pfx + '.attn.wv'].ref).reshape([-1, sl, nHead, headDim]);
    q = applyRoPE(q, rc.ref, rs.ref);
    k = applyRoPE(k, last ? rc : rc.ref, last ? rs : rs.ref);
    const attn = nn.dotProductAttention(q, k, v, { isCausal: true });
    x = x.add(np.dot(attn.reshape([-1, sl, nEmbd]), p[pfx + '.attn.wout'].ref));
    const n2 = rmsNorm(x.ref, p[pfx + '.norm2'].ref);
    let h = np.dot(n2, p[pfx + '.mlp.up'].ref);
    h = np.square(nn.relu(h)); // relu²
    h = np.dot(h, p[pfx + '.mlp.down'].ref);
    x = x.add(h);
  }
  x = rmsNorm(x, p['final_norm'].ref);
  return np.tanh(np.dot(x, p['unembed'].ref).mul(1/15)).mul(15); // softcap=15
}

// --- loss ---
function lossFn(p, input, target) {
  const logits = forward(p, input);
  const logProbs = nn.logSoftmax(logits, -1);
  const targets = nn.oneHot(target.reshape([-1]), VOCAB_SIZE);
  return logProbs.reshape([-1, VOCAB_SIZE]).mul(targets).sum().mul(-1 / (batchSize * seqLen));
}

// --- training loop ---
const optimizer = adamw((step) => {
  const progress = Math.min(elapsed / (trainSeconds * 1000), 1);
  return lr * lrSchedule(progress, warmupRatio, cooldownRatio);
}, { weightDecay, b1: 0.9, b2: 0.95 });

let optState = optimizer.init(tree.ref(params));
const lossGrad = jit(valueAndGrad((p, input, target) => lossFn(p, input, target)));

let step = 0, elapsed = 0;
const t0 = performance.now();

while (elapsed < trainSeconds * 1000 && !signal.aborted) {
  const batch = trainData.nextBatch(batchSize, seqLen);
  const [lossVal, grads] = lossGrad(tree.ref(params), batch.input, batch.target);
  const [updates, newState] = optimizer.update(grads, optState, tree.ref(params));
  optState = newState;
  for (const k in updates) { params[k] = applyUpdates({ [k]: params[k] }, { [k]: updates[k] })[k]; }
  await blockUntilReady(params);
  const loss = await lossVal.jsAsync();
  elapsed = performance.now() - t0;
  step++;
  onStep({ step, loss, elapsed });
  await yieldToUI();
  if (isNaN(loss)) break;
}

// --- evaluate & return ---
const valBpb = await evaluate(params, forward, VOCAB_SIZE, valData, batchSize, seqLen);
onReturn({ params, forward, vocabSize: VOCAB_SIZE, batchSize, seqLen, valBpb });
`;
