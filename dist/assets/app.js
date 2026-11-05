const state = {
  model: null,
  chunks: [],
  lastResult: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const elements = {
  form: $("#risk-form"),
  formError: $("#form-error"),
  resultEmpty: $("#result-empty"),
  resultContent: $("#result-content"),
  riskScore: $("#risk-score"),
  scoreRing: $("#score-ring"),
  scoreMarker: $("#score-marker"),
  riskBadge: $("#risk-badge"),
  riskTitle: $("#risk-title"),
  riskDescription: $("#risk-description"),
  recommendations: $("#recommendations"),
  chatForm: $("#chat-form"),
  chatQuestion: $("#chat-question"),
  chatMessages: $("#chat-messages"),
  toast: $("#toast"),
};

const modelPromise = fetch("./assets/model.json")
  .then((response) => {
    if (!response.ok) throw new Error("Không thể tải mô hình");
    return response.json();
  })
  .then((model) => {
    state.model = model;
    return model;
  })
  .catch((error) => {
    $(".topbar-status span:last-child").textContent = "Không tải được mô hình";
    throw error;
  });

const knowledgePromise = fetch("./assets/knowledge.json")
  .then((response) => (response.ok ? response.json() : []))
  .then((chunks) => {
    state.chunks = chunks;
    return chunks;
  })
  .catch(() => []);

function dense(input, layer) {
  const [inputSize, outputSize] = layer.shape;
  const output = new Array(outputSize);
  for (let outIndex = 0; outIndex < outputSize; outIndex += 1) {
    let sum = layer.bias[outIndex];
    for (let inputIndex = 0; inputIndex < inputSize; inputIndex += 1) {
      sum += input[inputIndex] * layer.kernel[inputIndex * outputSize + outIndex];
    }
    if (layer.activation === "relu") output[outIndex] = Math.max(0, sum);
    else if (layer.activation === "sigmoid") output[outIndex] = 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, sum))));
    else output[outIndex] = sum;
  }
  return output;
}

function batchNorm(input, layer) {
  return input.map(
    (value, index) =>
      layer.gamma[index] *
        ((value - layer.mean[index]) / Math.sqrt(layer.variance[index] + layer.epsilon)) +
      layer.beta[index],
  );
}

function preprocess(patient, model) {
  const prep = model.preprocessor;
  const numeric = prep.numericOrder.map(
    (name, index) => (Number(patient[name]) - prep.mean[index]) / prep.scale[index],
  );
  const gender = prep.gender.map((value) => (patient.gender === value ? 1 : 0));
  const smoking = prep.smoking.map((value) => (patient.smoking_history === value ? 1 : 0));
  const binary = prep.binaryOrder.map((name) => Number(patient[name]));
  return [...numeric, ...gender, ...smoking, ...binary];
}

function predictionAgent(patient, model) {
  let output = preprocess(patient, model);
  for (const layer of model.layers) {
    output = layer.type === "dense" ? dense(output, layer) : batchNorm(output, layer);
  }
  const riskScore = Math.max(0, Math.min(1, output[0]));
  return {
    agent: "PredictionAgent",
    riskScore,
    threshold: model.threshold,
    predictedLabel: Number(riskScore >= model.threshold),
    patient,
  };
}

function recommendationAgent(patient, prediction) {
  const items = [];
  let summary;
  if (prediction.predictedLabel === 1) {
    summary = "Mô hình ghi nhận tín hiệu nguy cơ cao.";
    items.push(
      "Nên đặt lịch kiểm tra y tế sớm để được đánh giá bằng xét nghiệm và thăm khám phù hợp.",
      "Theo dõi đường huyết, HbA1c và mang theo kết quả này khi trao đổi với nhân viên y tế.",
      "Không tự chẩn đoán hoặc tự thay đổi thuốc chỉ dựa trên điểm của mô hình.",
    );
  } else if (prediction.riskScore >= 0.4) {
    summary = "Điểm chưa vượt ngưỡng nhưng có tín hiệu cần theo dõi.";
    items.push(
      "Cân nhắc kiểm tra đường huyết định kỳ, đặc biệt khi có triệu chứng hoặc tiền sử gia đình.",
      "Duy trì vận động đều đặn, bữa ăn cân bằng và theo dõi cân nặng.",
      "Trao đổi với nhân viên y tế nếu các chỉ số tiếp tục tăng.",
    );
  } else {
    summary = "Mô hình chưa ghi nhận nguy cơ cao tại thời điểm này.";
    items.push(
      "Tiếp tục duy trì vận động đều đặn, bữa ăn cân bằng và cân nặng phù hợp.",
      "Theo dõi sức khỏe định kỳ; kết quả thấp không loại trừ hoàn toàn bệnh tiểu đường.",
      "Đi khám nếu xuất hiện khát nhiều, tiểu nhiều, sụt cân không rõ nguyên nhân hoặc mệt mỏi kéo dài.",
    );
  }

  if (Number(patient.HbA1c_level) >= 6.5) {
    items.splice(1, 0, "HbA1c nhập vào đang ở mức đáng chú ý; nên xác nhận bằng cơ sở y tế thay vì tự kết luận.");
  } else if (Number(patient.HbA1c_level) >= 5.7) {
    items.splice(1, 0, "HbA1c nhập vào cao hơn vùng thường gặp; nên theo dõi và trao đổi khi khám định kỳ.");
  }
  if (Number(patient.bmi) >= 30) {
    items.push("Có thể trao đổi về kế hoạch kiểm soát cân nặng an toàn, phù hợp với thể trạng cá nhân.");
  }
  if (Number(patient.hypertension) === 1 || Number(patient.heart_disease) === 1) {
    items.push("Tiền sử tim mạch/huyết áp nên được xem xét cùng toàn bộ hồ sơ sức khỏe khi đánh giá nguy cơ.");
  }

  return { agent: "RecommendationAgent", summary, items: items.slice(0, 5) };
}

function coordinatorAgent(patient, model) {
  const prediction = predictionAgent(patient, model);
  const recommendation = recommendationAgent(patient, prediction);
  return { taskType: "full_assessment", prediction, recommendation };
}

function readPatient(form) {
  const data = new FormData(form);
  return {
    gender: data.get("gender"),
    age: Number(data.get("age")),
    hypertension: Number(data.get("hypertension")),
    heart_disease: Number(data.get("heart_disease")),
    smoking_history: data.get("smoking_history"),
    bmi: Number(data.get("bmi")),
    HbA1c_level: Number(data.get("HbA1c_level")),
    blood_glucose_level: Number(data.get("blood_glucose_level")),
  };
}

function validatePatient(patient) {
  const checks = [
    [patient.age >= 0 && patient.age <= 100, "Tuổi cần nằm trong khoảng 0–100."],
    [patient.bmi >= 10 && patient.bmi <= 80, "BMI cần nằm trong khoảng 10–80 kg/m²."],
    [patient.HbA1c_level >= 3 && patient.HbA1c_level <= 20, "HbA1c cần nằm trong khoảng 3–20%."],
    [patient.blood_glucose_level >= 50 && patient.blood_glucose_level <= 400, "Đường huyết cần nằm trong khoảng 50–400 mg/dL."],
  ];
  return checks.find(([valid]) => !valid)?.[1] || "";
}

function riskPresentation(score, threshold) {
  if (score >= threshold) {
    return {
      band: "high",
      badge: "Nguy cơ cao",
      title: "Nên kiểm tra y tế sớm",
      description: "Điểm mô hình vượt ngưỡng phân loại đã tối ưu trên tập validation. Đây là tín hiệu sàng lọc, không phải chẩn đoán.",
    };
  }
  if (score >= 0.4) {
    return {
      band: "medium",
      badge: "Cần theo dõi",
      title: "Chưa vượt ngưỡng",
      description: "Điểm mô hình chưa đạt ngưỡng phân loại nhưng một số chỉ số có thể cần được theo dõi trong bối cảnh sức khỏe tổng thể.",
    };
  }
  return {
    band: "low",
    badge: "Nguy cơ thấp",
    title: "Chưa thấy tín hiệu rõ",
    description: "Điểm mô hình ở dưới ngưỡng phân loại. Hãy tiếp tục duy trì lối sống lành mạnh và khám định kỳ.",
  };
}

function renderResult(result) {
  const { prediction, recommendation } = result;
  const scorePercent = prediction.riskScore * 100;
  const presentation = riskPresentation(prediction.riskScore, prediction.threshold);

  elements.resultEmpty.hidden = true;
  elements.resultContent.hidden = false;
  elements.riskScore.textContent = `${scorePercent.toFixed(1)}%`;
  elements.scoreRing.style.setProperty("--score", scorePercent.toFixed(2));
  elements.scoreMarker.style.left = `${scorePercent}%`;
  elements.riskBadge.textContent = presentation.badge;
  elements.riskBadge.className = `risk-badge ${presentation.band}`;
  elements.riskTitle.textContent = presentation.title;
  elements.riskDescription.textContent = presentation.description;
  elements.recommendations.innerHTML = recommendation.items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  $("#agent-timestamp").textContent = new Intl.DateTimeFormat("vi-VN", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date());
}

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const patient = readPatient(elements.form);
  const error = validatePatient(patient);
  elements.formError.textContent = error;
  if (error) return;

  const submit = $("button[type='submit']", elements.form);
  const original = submit.innerHTML;
  submit.disabled = true;
  submit.textContent = "Đang điều phối các agent…";
  try {
    const model = await modelPromise;
    await new Promise((resolve) => setTimeout(resolve, 360));
    const result = coordinatorAgent(patient, model);
    state.lastResult = result;
    renderResult(result);
    if (window.innerWidth < 900) $("#result-card").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (errorObject) {
    elements.formError.textContent = "Không thể chạy mô hình. Hãy tải lại trang và thử lại.";
    console.error(errorObject);
  } finally {
    submit.disabled = false;
    submit.innerHTML = original;
  }
});

$("#sample-data").addEventListener("click", () => {
  const sample = {
    gender: "Female",
    age: 54,
    smoking_history: "former",
    bmi: 31.2,
    HbA1c_level: 6.8,
    blood_glucose_level: 185,
  };
  Object.entries(sample).forEach(([name, value]) => {
    elements.form.elements[name].value = value;
  });
  $("#hypertension-yes").checked = true;
  $("#heart-no").checked = true;
  showToast("Đã điền ca mẫu từ Untitled2.ipynb");
});

$("#reset-result").addEventListener("click", () => {
  elements.resultContent.hidden = true;
  elements.resultEmpty.hidden = false;
  state.lastResult = null;
});

$("#copy-result").addEventListener("click", async () => {
  if (!state.lastResult) return;
  const { prediction, recommendation } = state.lastResult;
  const text = [
    "KẾT QUẢ SÀNG LỌC NGUY CƠ TIỂU ĐƯỜNG",
    `Điểm mô hình: ${(prediction.riskScore * 100).toFixed(1)}%`,
    `Ngưỡng phân loại: ${(prediction.threshold * 100).toFixed(2)}%`,
    recommendation.summary,
    ...recommendation.items.map((item, index) => `${index + 1}. ${item}`),
    "Lưu ý: Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán y khoa.",
  ].join("\n");
  try {
    await navigator.clipboard.writeText(text);
    showToast("Đã sao chép kết quả");
  } catch {
    showToast("Trình duyệt không cho phép sao chép tự động");
  }
});

const topicDefinitions = [
  {
    id: "symptoms",
    keywords: ["trieu chung", "dau hieu", "symptom", "khat", "tieu nhieu", "met moi", "sut can"],
    expansion: "symptoms thirsty hungry urinating tired numbness tingling sores heal losing weight",
    answer:
      "Các biểu hiện có thể gặp gồm khát hoặc đói nhiều, đi tiểu thường xuyên, mệt mỏi, sụt cân không chủ ý, tê hoặc châm chích tay chân và vết thương lâu lành. Với tiểu đường type 2, triệu chứng có thể tiến triển chậm hoặc rất nhẹ nên nhiều người không nhận ra. Nếu có triệu chứng kéo dài, bạn nên trao đổi với nhân viên y tế và làm xét nghiệm phù hợp.",
  },
  {
    id: "hba1c",
    keywords: ["hba1c", "a1c", "duong trung binh"],
    expansion: "A1C test blood glucose diagnosis average months",
    answer:
      "HbA1c (hay A1C) là xét nghiệm phản ánh mức đường huyết trung bình trong khoảng vài tháng gần đây. Nhân viên y tế có thể dùng A1C cùng các xét nghiệm đường huyết khác để đánh giá và theo dõi bệnh tiểu đường. Một kết quả đơn lẻ cần được diễn giải theo bối cảnh sức khỏe và hướng dẫn chuyên môn.",
  },
  {
    id: "prevention",
    keywords: ["phong ngua", "phong tranh", "ngan ngua", "prevent", "loi song", "van dong"],
    expansion: "prevent prevention healthy diet healthy weight physical activity lifestyle",
    answer:
      "Tiểu đường type 1 hiện không có cách phòng ngừa đã được xác lập. Với type 2, duy trì cân nặng phù hợp, vận động đều đặn và ăn uống cân bằng có thể giúp trì hoãn hoặc giảm nguy cơ. Kế hoạch nên vừa sức và được điều chỉnh nếu bạn có bệnh nền hoặc đang mang thai.",
  },
  {
    id: "tests",
    keywords: ["xet nghiem", "chan doan", "kiem tra", "test", "diagnos", "duong huyet"],
    expansion: "diagnosed glucose blood tests A1C test provider screening",
    answer:
      "Chẩn đoán cần dựa trên xét nghiệm, không chỉ triệu chứng hay điểm dự đoán. Cơ sở y tế có thể sử dụng một hoặc nhiều xét nghiệm đường huyết, trong đó có A1C. Loại xét nghiệm và việc cần xác nhận lại kết quả phụ thuộc vào tình huống cụ thể.",
  },
  {
    id: "risk",
    keywords: ["nguy co", "nguyen nhan", "ai de mac", "risk", "cause", "thua can", "gia dinh"],
    expansion: "risk factors causes overweight obesity age family prediabetes gestational high blood pressure inactive",
    answer:
      "Nguy cơ type 2 có thể liên quan đến thừa cân, ít vận động, tuổi, tiền sử gia đình, tiền tiểu đường, tăng huyết áp hoặc từng bị tiểu đường thai kỳ. Không phải có một yếu tố là chắc chắn mắc bệnh; đánh giá cần xem tổng thể nhiều yếu tố và kết quả xét nghiệm.",
  },
  {
    id: "treatment",
    keywords: ["dieu tri", "thuoc", "insulin", "treatment", "chua"],
    expansion: "treatment manage insulin medicine healthy diet physical activity blood glucose",
    answer:
      "Điều trị phụ thuộc loại tiểu đường và từng người. Type 1 cần insulin hằng ngày; type 2 có thể được quản lý bằng thay đổi lối sống và đôi khi cần thuốc. Theo dõi đường huyết là một phần quan trọng, nhưng phác đồ cụ thể phải do nhân viên y tế hướng dẫn.",
  },
  {
    id: "types",
    keywords: ["cac loai", "type 1", "type 2", "thai ky", "gestational"],
    expansion: "types type 1 type 2 gestational insulin pregnancy",
    answer:
      "Ba nhóm thường được nhắc đến là type 1, type 2 và tiểu đường thai kỳ. Type 1 liên quan đến việc cơ thể tạo rất ít hoặc không tạo insulin; type 2 thường liên quan đến đáp ứng insulin kém; tiểu đường thai kỳ xuất hiện trong thai kỳ và cần được theo dõi chuyên môn.",
  },
  {
    id: "urgent",
    keywords: ["cap cuu", "nguy hiem", "non", "tho nhanh", "emergency"],
    expansion: "nausea vomiting rapid breathing very high blood glucose emergency",
    answer:
      "Buồn nôn, nôn, thở nhanh, lú lẫn hoặc tình trạng xấu đi nhanh kèm đường huyết rất cao có thể cần được đánh giá khẩn cấp. Nếu bạn hoặc người bên cạnh đang có dấu hiệu nặng, hãy liên hệ dịch vụ cấp cứu tại nơi bạn sống thay vì chờ công cụ trực tuyến.",
  },
];

function normalize(text) {
  return text
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokens(text) {
  const stop = new Set(["the", "and", "or", "a", "an", "is", "are", "of", "to", "in", "for", "with", "what", "how"]);
  return normalize(text)
    .split(" ")
    .filter((token) => token.length > 2 && !stop.has(token));
}

function chooseTopic(question) {
  const normalized = normalize(question);
  let best = topicDefinitions[0];
  let bestScore = 0;
  for (const topic of topicDefinitions) {
    const score = topic.keywords.reduce((sum, keyword) => sum + (normalized.includes(normalize(keyword)) ? 1 : 0), 0);
    if (score > bestScore) {
      best = topic;
      bestScore = score;
    }
  }
  return bestScore ? best : null;
}

function retrieveSources(question, topic, topK = 3) {
  const queryTokens = new Set(tokens(`${question} ${topic?.expansion || "diabetes health"}`));
  const ranked = state.chunks
    .map((chunk) => {
      const textTokens = tokens(chunk.text);
      const unique = new Set(textTokens);
      let overlap = 0;
      queryTokens.forEach((token) => {
        if (unique.has(token)) overlap += token.length > 7 ? 1.6 : 1;
      });
      return { ...chunk, score: overlap / Math.sqrt(Math.max(1, unique.size)) };
    })
    .sort((a, b) => b.score - a.score);

  const uniqueSources = [];
  for (const item of ranked) {
    if (!uniqueSources.some((source) => source.source_url === item.source_url)) uniqueSources.push(item);
    if (uniqueSources.length >= topK) break;
  }
  return uniqueSources;
}

function knowledgeAgent(question) {
  const topic = chooseTopic(question);
  const sources = retrieveSources(question, topic);
  const answer = topic
    ? topic.answer
    : "Mình chưa nhận diện được chủ đề đủ rõ để trả lời chắc chắn. Bạn có thể hỏi cụ thể hơn về triệu chứng, HbA1c, xét nghiệm, yếu tố nguy cơ, phòng ngừa hoặc điều trị. Các nguồn phù hợp nhất với câu hỏi hiện tại được đính kèm bên dưới.";
  return { agent: "KnowledgeAgent", question, answer, sources };
}

function addMessage(role, content, sources = []) {
  const message = document.createElement("div");
  message.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;
  const avatar = role === "user" ? "B" : "K";
  message.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div>
      <p>${escapeHtml(content)}</p>
      ${
        sources.length
          ? `<div class="answer-sources">${sources
              .map(
                (source, index) =>
                  `<a href="${escapeAttribute(source.source_url)}" target="_blank" rel="noreferrer">[${index + 1}] ${escapeHtml(shortSource(source.title))}</a>`,
              )
              .join("")}</div>`
          : ""
      }
      <span class="message-meta">${role === "user" ? "Bạn" : `Knowledge Agent · ${sources.length || 3} nguồn`}</span>
    </div>`;
  elements.chatMessages.append(message);
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function addTyping() {
  const typing = document.createElement("div");
  typing.className = "message assistant-message typing";
  typing.innerHTML = '<div class="message-avatar">K</div><div><p><i></i><i></i><i></i></p></div>';
  elements.chatMessages.append(typing);
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
  return typing;
}

elements.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = elements.chatQuestion.value.trim();
  if (!question) return;
  addMessage("user", question);
  elements.chatQuestion.value = "";
  elements.chatQuestion.style.height = "auto";
  const typing = addTyping();
  await knowledgePromise;
  await new Promise((resolve) => setTimeout(resolve, 520));
  const result = knowledgeAgent(question);
  typing.remove();
  addMessage("assistant", result.answer, result.sources);
});

$("#suggestion-row").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  elements.chatQuestion.value = button.textContent;
  elements.chatForm.requestSubmit();
});

elements.chatQuestion.addEventListener("input", () => {
  elements.chatQuestion.style.height = "auto";
  elements.chatQuestion.style.height = `${Math.min(elements.chatQuestion.scrollHeight, 100)}px`;
});

function shortSource(title) {
  if (/WHO/i.test(title)) return "WHO";
  if (/NHS/i.test(title)) return "NHS";
  return "MedlinePlus";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => elements.toast.classList.remove("show"), 2200);
}

const menuButton = $(".menu-button");
menuButton.addEventListener("click", () => {
  const open = document.body.classList.toggle("menu-open");
  menuButton.setAttribute("aria-expanded", String(open));
});
$$('.nav-link').forEach((link) => link.addEventListener("click", () => document.body.classList.remove("menu-open")));

const observer = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    $$(".nav-link").forEach((link) => link.classList.toggle("active", link.dataset.section === visible.target.id));
  },
  { rootMargin: "-25% 0px -55% 0px", threshold: [0.01, 0.2, 0.5] },
);
[$("#assessment"), $("#knowledge"), $("#model")].forEach((section) => observer.observe(section));

$$('[data-modal-open]').forEach((button) => {
  button.addEventListener("click", () => $(`#${button.dataset.modalOpen}`).showModal());
});
$$('[data-modal-close]').forEach((button) => button.addEventListener("click", () => button.closest("dialog").close()));

$$('.plot-toggle').forEach((button) => {
  button.addEventListener("click", () => {
    $$(".plot-toggle").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    const image = $("#model-plot");
    if (button.dataset.plot === "matrix") {
      image.src = "./assets/confusion-matrix.png";
      image.alt = "Ma trận nhầm lẫn của mô hình";
    } else {
      image.src = "./assets/evaluation-curves.png";
      image.alt = "Biểu đồ đường cong ROC và Precision-Recall của mô hình";
    }
  });
});

window.__diabetesAI = {
  ready: Promise.all([modelPromise, knowledgePromise]),
  predict: async (patient) => predictionAgent(patient, await modelPromise),
  ask: async (question) => {
    await knowledgePromise;
    return knowledgeAgent(question);
  },
};
