/*
 * lm_engine.js — 纯前端语言模型推理引擎（rnn / lstm / transformer 通用）
 *
 * 逐位复现 language_model.py 里三种结构的前向：
 *   rnn/lstm     : Embedding → N × RNN/LSTM → Linear(hidden → vocab)
 *   transformer  : Embedding + 位置编码
 *                  → N × [ LayerNorm → 因果多头自注意力 → 残差 → LayerNorm → FFN(ReLU) → 残差 ]
 *                  → Linear(embed_dim → vocab)
 *
 * 权重来自 tools/export_lm_model.py 导出的 lm_model_*.js（fp16 + base64），
 * 所有模型都注册在 window.LM_MODELS[<arch>] 上，可以同时加载多个用于对比。
 *
 * 浏览器里：<script src="lm_model_lstm.js"></script><script src="lm_engine.js"></script>
 * Node 里（对拍自检）：global.window = globalThis; require('./lm_model_lstm.js'); require('./lm_engine.js')
 */
(function (root) {
  "use strict";

  // ─────────────────────────── 基础工具 ───────────────────────────

  function b64ToBinary(b64) {
    if (typeof atob === "function") return atob(b64);
    return Buffer.from(b64, "base64").toString("binary");
  }

  // IEEE754 半精度 → 单精度
  function halfToFloat(h) {
    const s = h & 0x8000 ? -1 : 1;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function decodeFp16(b64, count) {
    const bin = b64ToBinary(b64);
    if (bin.length !== count * 2) {
      throw new Error(`权重长度不符: 期望 ${count * 2} 字节, 实际 ${bin.length}`);
    }
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      out[i] = halfToFloat(bin.charCodeAt(i * 2) | (bin.charCodeAt(i * 2 + 1) << 8));
    }
    return out;
  }

  // ─────────────────────────── 算子 ───────────────────────────

  // y = W[r0:r0+outDim] · x + b[r0:r0+outDim]，W 按 (out, in) 行主序存放
  function linear(W, b, x, outDim, inDim, rowOffset) {
    const r0 = rowOffset || 0;
    const y = new Float32Array(outDim);
    for (let o = 0; o < outDim; o++) {
      const base = (r0 + o) * inDim;
      let s = b ? b[r0 + o] : 0;
      for (let i = 0; i < inDim; i++) s += W[base + i] * x[i];
      y[o] = s;
    }
    return y;
  }

  // LayerNorm：PyTorch 用有偏方差（除以 n）
  function layerNorm(v, w, b, n, eps) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += v[i];
    mean /= n;
    let varr = 0;
    for (let i = 0; i < n; i++) {
      const d = v[i] - mean;
      varr += d * d;
    }
    varr /= n;
    const inv = 1 / Math.sqrt(varr + eps);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = (v[i] - mean) * inv * w[i] + b[i];
    return out;
  }

  function logSoftmaxArr(logits, temperature) {
    const t = temperature && temperature > 0 ? temperature : 1;
    const n = logits.length;
    let max = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = logits[i] / t;
      if (v > max) max = v;
    }
    let sum = 0;
    for (let i = 0; i < n; i++) sum += Math.exp(logits[i] / t - max);
    const logZ = max + Math.log(sum);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = logits[i] / t - logZ;
    return out;
  }

  // 取最大的 k 个下标（k 很小，直接插排）
  function topK(arr, k) {
    const idx = [];
    for (let i = 0; i < arr.length; i++) {
      if (idx.length < k) {
        idx.push(i);
        idx.sort((a, b) => arr[b] - arr[a]);
      } else if (arr[i] > arr[idx[idx.length - 1]]) {
        idx[idx.length - 1] = i;
        idx.sort((a, b) => arr[b] - arr[a]);
      }
    }
    return idx;
  }

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  // ─────────────────────────── 模型 ───────────────────────────

  class CharLM {
    constructor(data) {
      const meta = data.meta;
      this.meta = meta;
      this.arch = meta.arch;              // "rnn" | "lstm" | "transformer"
      this.V = meta.vocab_size;
      this.E = meta.embed_dim;
      this.H = meta.hidden_dim;
      this.L = meta.num_layers;
      this.eps = meta.norm_eps;

      this.vocab = Array.from(data.vocab);
      this.char2id = new Map();
      for (let i = 0; i < this.vocab.length; i++) this.char2id.set(this.vocab[i], i);

      const w = data.weights;
      this.embed = decodeFp16(w.embed, this.V * this.E);
      this.fcW = decodeFp16(w.fc_w, this.V * meta.fc_in);
      this.fcB = decodeFp16(w.fc_b, this.V);

      if (this.arch === "transformer") {
        this.NH = meta.nhead;
        this.HD = this.E / this.NH;
        this.F = this.H;
        this.maxLen = meta.max_len;
        this.pos = decodeFp16(w.pos, this.maxLen * this.E);
        this.layers = w.layers.map((l) => ({
          qkvW: decodeFp16(l.qkv_w, 3 * this.E * this.E),
          qkvB: decodeFp16(l.qkv_b, 3 * this.E),
          outW: decodeFp16(l.out_w, this.E * this.E),
          outB: decodeFp16(l.out_b, this.E),
          l1W: decodeFp16(l.l1_w, this.F * this.E),
          l1B: decodeFp16(l.l1_b, this.F),
          l2W: decodeFp16(l.l2_w, this.E * this.F),
          l2B: decodeFp16(l.l2_b, this.E),
          n1W: decodeFp16(l.n1_w, this.E),
          n1B: decodeFp16(l.n1_b, this.E),
          n2W: decodeFp16(l.n2_w, this.E),
          n2B: decodeFp16(l.n2_b, this.E),
        }));
      } else {
        this.maxLen = 0;                  // RNN 类没有位置上限
        this.layers = w.layers.map((l) => {
          const rows = l.gates * l.hidden;
          return {
            wIh: decodeFp16(l.w_ih, rows * l.in),
            wHh: decodeFp16(l.w_hh, rows * l.hidden),
            bias: decodeFp16(l.bias, rows),
            in: l.in,
            hidden: l.hidden,
            gates: l.gates,
            rows: rows,
          };
        });
      }
      this.loaded = true;
    }

    ids(text) {
      const out = [];
      for (const c of Array.from(text)) {
        const id = this.char2id.get(c);
        if (id !== undefined) out.push(id);
      }
      return out;
    }

    // 返回最后一个位置的 logits（长度 V）
    logits(ids) {
      if (!ids.length) throw new Error("上下文为空");
      return this.arch === "transformer" ? this._logitsTransformer(ids) : this._logitsRNN(ids);
    }

    // ── RNN / LSTM ──
    _logitsRNN(ids) {
      const T = ids.length, E = this.E, H = this.H, L = this.L;
      const lstm = this.arch === "lstm";
      const hs = [], cs = [];
      for (let i = 0; i < L; i++) {
        hs.push(new Float32Array(H));
        cs.push(new Float32Array(H));
      }

      let inp = null;
      for (let t = 0; t < T; t++) {
        const eb = ids[t] * E;
        inp = this.embed.subarray(eb, eb + E);
        for (let li = 0; li < L; li++) {
          const p = this.layers[li];
          const h = hs[li], c = cs[li];
          const g = new Float32Array(p.rows);
          for (let o = 0; o < p.rows; o++) {
            const ri = o * p.in, rh = o * H;
            let s = p.bias[o];
            for (let i = 0; i < p.in; i++) s += p.wIh[ri + i] * inp[i];
            for (let i = 0; i < H; i++) s += p.wHh[rh + i] * h[i];
            g[o] = s;
          }
          if (lstm) {
            // PyTorch 的门顺序是 i, f, g, o
            for (let j = 0; j < H; j++) {
              const ig = sigmoid(g[j]);
              const fg = sigmoid(g[H + j]);
              const gg = Math.tanh(g[2 * H + j]);
              const og = sigmoid(g[3 * H + j]);
              c[j] = fg * c[j] + ig * gg;
              h[j] = og * Math.tanh(c[j]);
            }
          } else {
            for (let j = 0; j < H; j++) h[j] = Math.tanh(g[j]);
          }
          inp = h;
        }
      }
      return linear(this.fcW, this.fcB, inp, this.V, this.H, 0);
    }

    // ── Transformer ──
    _logitsTransformer(ids) {
      const T = ids.length;
      const E = this.E, F = this.F, NH = this.NH, HD = this.HD;
      if (T > this.maxLen) throw new Error(`上下文 ${T} 字超过位置编码上限 ${this.maxLen}`);

      const x = [];
      for (let t = 0; t < T; t++) {
        const eb = ids[t] * E, pb = t * E;
        const v = new Float32Array(E);
        for (let i = 0; i < E; i++) v[i] = this.embed[eb + i] + this.pos[pb + i];
        x.push(v);
      }

      const scale = 1 / Math.sqrt(HD);
      for (const L of this.layers) {
        // Pre-LN：注意力在归一化后的输入上算，结果加回原输入
        const q = [], k = [], v = [];
        for (let t = 0; t < T; t++) {
          const h = layerNorm(x[t], L.n1W, L.n1B, E, this.eps);
          q.push(linear(L.qkvW, L.qkvB, h, E, E, 0));
          k.push(linear(L.qkvW, L.qkvB, h, E, E, E));
          v.push(linear(L.qkvW, L.qkvB, h, E, E, 2 * E));
        }

        const scores = new Float32Array(T);
        for (let t = 0; t < T; t++) {
          const att = new Float32Array(E);
          for (let hi = 0; hi < NH; hi++) {
            const hb = hi * HD;
            let maxS = -Infinity;
            for (let s = 0; s <= t; s++) {          // 因果：只看 s <= t
              let d = 0;
              for (let j = 0; j < HD; j++) d += q[t][hb + j] * k[s][hb + j];
              d *= scale;
              scores[s] = d;
              if (d > maxS) maxS = d;
            }
            let sum = 0;
            for (let s = 0; s <= t; s++) {
              scores[s] = Math.exp(scores[s] - maxS);
              sum += scores[s];
            }
            for (let j = 0; j < HD; j++) {
              let acc = 0;
              for (let s = 0; s <= t; s++) acc += (scores[s] / sum) * v[s][hb + j];
              att[hb + j] = acc;
            }
          }
          const proj = linear(L.outW, L.outB, att, E, E, 0);
          for (let i = 0; i < E; i++) x[t][i] += proj[i];        // 残差
        }

        for (let t = 0; t < T; t++) {
          const h2 = layerNorm(x[t], L.n2W, L.n2B, E, this.eps);
          const f1 = linear(L.l1W, L.l1B, h2, F, E, 0);
          for (let i = 0; i < F; i++) if (f1[i] < 0) f1[i] = 0;  // ReLU
          const f2 = linear(L.l2W, L.l2B, f1, E, F, 0);
          for (let i = 0; i < E; i++) x[t][i] += f2[i];          // 残差
        }
      }
      return linear(this.fcW, this.fcB, x[T - 1], this.V, E, 0);
    }

    // 下一个字的分布；entries 按概率降序
    nextDist(ids, k, temperature) {
      const logp = logSoftmaxArr(this.logits(ids), temperature);
      const idx = topK(logp, k || 10);
      return {
        entries: idx.map((i) => ({ char: this.vocab[i], logp: logp[i], p: Math.exp(logp[i]) })),
        logp,
      };
    }

    predict(text, k, temperature) {
      const ids = this.ids(text);
      return { ids, dist: this.nextDist(ids, k, temperature) };
    }

    // 贪心续写：每步取概率最大的字
    generate(text, steps, temperature) {
      const ids = this.ids(text);
      const added = [];
      for (let i = 0; i < steps; i++) {
        const d = this.nextDist(ids.concat(added), 1, temperature);
        added.push(this.char2id.get(d.entries[0].char));
        if (this.arch === "transformer" && ids.length + added.length >= this.maxLen) break;
      }
      return { ids, added, text: added.map((i) => this.vocab[i]).join("") };
    }
  }

  root.LM_ENGINE = { CharLM, decodeFp16, halfToFloat, logSoftmaxArr, topK };
})(typeof window !== "undefined" ? window : globalThis);
