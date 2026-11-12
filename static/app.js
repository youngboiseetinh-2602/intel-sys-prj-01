const form = document.querySelector('#patient-form');
const button = document.querySelector('#submit');
const error = document.querySelector('#error');
let controller;
function clearResult() {
  document.querySelector('#result').hidden = true;
  document.querySelector('#empty').hidden = false;
  error.hidden = true;
}
form.addEventListener('input', () => { controller?.abort(); clearResult(); });
document.querySelector('#sample').addEventListener('click', () => {
  controller?.abort();
  const sample = {gender:'Female',age:54,bmi:31.2,HbA1c_level:6.8,blood_glucose_level:185,smoking_history:'former',hypertension:1,heart_disease:0};
  for (const [key, value] of Object.entries(sample)) form.elements.namedItem(key).value = value;
  clearResult();
});
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  clearResult();
  const patient = Object.fromEntries(new FormData(form));
  for (const name of ['age','bmi','HbA1c_level','blood_glucose_level','hypertension','heart_disease']) patient[name] = Number(patient[name]);
  controller = new AbortController();
  button.disabled = true;
  button.textContent = 'Đang chạy mô hình…';
  try {
    const response = await fetch('/api/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(patient),signal:controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Vui lòng kiểm tra các trường thông tin.');
    if (!Number.isFinite(data.score) || data.score < 0 || data.score > 1 || ![0,1].includes(data.predicted_label)) throw new Error('Phản hồi mô hình không hợp lệ.');
    document.querySelector('#empty').hidden = true;
    document.querySelector('#result').hidden = false;
    document.querySelector('#label').textContent = data.predicted_label ? 'NHÃN 1 · DƯƠNG TÍNH THEO MÔ HÌNH' : 'NHÃN 0 · ÂM TÍNH THEO MÔ HÌNH';
    document.querySelector('#score').textContent = data.score.toFixed(4);
    document.querySelector('#bar').style.width = `${data.score * 100}%`;
    document.querySelector('#threshold').textContent = `Ngưỡng phân loại: ${data.threshold.toFixed(7)} · ${data.model_version}`;
    document.querySelector('#explanation').textContent = data.predicted_label ? 'Điểm đạt hoặc vượt ngưỡng phân loại của mô hình.' : 'Điểm thấp hơn ngưỡng phân loại của mô hình.';
  } catch (exc) {
    if (exc.name !== 'AbortError') {
      error.textContent = exc instanceof TypeError ? 'Không kết nối được máy chủ. Vui lòng thử lại.' : exc.message;
      error.hidden = false;
    }
  } finally {
    button.disabled = false;
    button.innerHTML = 'Xem kết quả phân loại <span>↗</span>';
  }
});
