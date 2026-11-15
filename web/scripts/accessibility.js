// Persistent display preferences contain no patient data or credentials.
const KEY = "behaviorsense.accessibility";
const labels = {
  en:["Display preferences","Text size","Standard","Large","Extra large","High contrast","Reduce motion","Skip to main content"],
  hi:["देखने की सुविधाएँ","अक्षरों का आकार","सामान्य","बड़ा","बहुत बड़ा","अधिक कंट्रास्ट","हलचल कम करें","मुख्य सामग्री पर जाएँ"],
  mr:["दृश्य सुविधा","अक्षरांचा आकार","सामान्य","मोठा","खूप मोठा","अधिक कॉन्ट्रास्ट","हालचाल कमी करा","मुख्य मजकुराकडे जा"]
};
let settings = {size:"standard",contrast:false,motion:false};
try { settings = {...settings,...JSON.parse(localStorage.getItem(KEY)||"{}")} } catch {}
function apply() {
  const root=document.documentElement;
  root.dataset.textSize = ["standard","large","extra"].includes(settings.size) ? settings.size : "standard";
  root.dataset.highContrast=String(Boolean(settings.contrast));
  root.dataset.reduceMotion=String(Boolean(settings.motion));
  try { localStorage.setItem(KEY,JSON.stringify(settings)) } catch {}
}
export function mountAccessibility(language="en") {
  let host=document.getElementById("accessibility-controls");
  if (!host) { host=document.createElement("div");host.id="accessibility-controls";document.querySelector("header")?.after(host); }
  const text=labels[language]||labels.en;
  host.innerHTML=`<div class="a11y-toolbar" role="group" aria-label="${text[0]}"><label for="a11y-size">${text[1]} <select id="a11y-size"><option value="standard">${text[2]}</option><option value="large">${text[3]}</option><option value="extra">${text[4]}</option></select></label><label for="a11y-contrast"><input type="checkbox" id="a11y-contrast">${text[5]}</label><label for="a11y-motion"><input type="checkbox" id="a11y-motion">${text[6]}</label><a href="#${document.querySelector("main")?.id||"main-content"}">${text[7]}</a></div>`;
  document.getElementById("a11y-size").value=settings.size;
  document.getElementById("a11y-contrast").checked=Boolean(settings.contrast);
  document.getElementById("a11y-motion").checked=Boolean(settings.motion);
  host.onchange=()=> { settings={size:document.getElementById("a11y-size").value,contrast:document.getElementById("a11y-contrast").checked,motion:document.getElementById("a11y-motion").checked};apply(); };
  const main=document.querySelector("main");
  if(main&&!main.id)main.id="main-content";
  if(main)main.tabIndex=-1;
  for(const skip of document.querySelectorAll(".skip,.a11y-toolbar a"))skip.onclick=()=> {main?.focus();};
  apply();
}
if(!document.body.classList.contains("care-page")) {
  if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",()=>mountAccessibility(),{once:true});
  else mountAccessibility();
}
