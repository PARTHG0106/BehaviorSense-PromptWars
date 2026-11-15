/* Real browser accessibility and care-flow checks; only fictional records and preview mail. */
import {chromium} from "playwright-core";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import net from "node:net";
import {spawn} from "node:child_process";
import {fileURLToPath} from "node:url";
import assert from "node:assert/strict";

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),"..");
const temporary=await fs.mkdtemp(path.join(os.tmpdir(),"behaviorsense-ui-"));
const port=await new Promise(resolve=>{const probe=net.createServer();probe.listen(0,"127.0.0.1",()=>{const value=probe.address().port;probe.close(()=>resolve(value));});});
const base=`http://127.0.0.1:${port}`;
const server=spawn(process.env.PYTHON_EXECUTABLE||"python",[path.join(root,"web/care_backend.py"),"--port",String(port),"--database",path.join(temporary,"care.sqlite3")],{cwd:root,env:{...process.env,BS_MAIL_MODE:"preview"},windowsHide:true,stdio:["ignore","pipe","pipe"]});
let logs="";server.stdout.on("data",data=>{logs=(logs+data).slice(-8000)});server.stderr.on("data",data=>{logs=(logs+data).slice(-8000)});
let browser;
const audits=[];
try {
  let ready=false;
  for(let i=0;i<100;i++){try{const response=await fetch(base+"/api/care/health");if(response.ok){ready=true;break;}}catch{}await new Promise(resolve=>setTimeout(resolve,100));}
  assert(ready,logs||"Care server did not start");
  const executable=process.env.BROWSER_EXECUTABLE||(process.platform==="win32"?"C:/Program Files/Google/Chrome/Application/chrome.exe":undefined);
  browser=await chromium.launch({headless:true,...(executable?{executablePath:executable}:{})});
  const context=await browser.newContext({viewport:{width:1280,height:900}});
  // Developer instrumentation executes before page scripts; application CSP stays enabled.
  await context.addInitScript({path:path.join(root,"node_modules/axe-core/axe.min.js")});
  const page=await context.newPage();const errors=[];page.on("pageerror",error=>errors.push(error.message));
  async function audit(label){const result=await page.evaluate(async()=>await axe.run(document,{runOnly:{type:"tag",values:["wcag2a","wcag2aa","wcag21a","wcag21aa","wcag22aa"]}}));audits.push({label,violations:result.violations.map(v=>({id:v.id,impact:v.impact,help:v.help,targets:v.nodes.map(n=>n.target)}))});assert.equal(result.violations.length,0,JSON.stringify(audits.at(-1)));}
  await page.goto(base,{waitUntil:"networkidle"});await audit("care sign-in desktop");
  await page.keyboard.press("Tab");assert.match(await page.evaluate(()=>document.activeElement.textContent),/Skip/);
  await page.keyboard.press("Enter");assert.equal(await page.evaluate(()=>document.activeElement.id),"care-main");
  await page.getByRole("button",{name:"Try a fictional care profile",exact:true}).click();
  await page.locator("#care-workspace").waitFor({state:"visible"});await page.getByRole("button",{name:/Confirm taken: Prescribed/}).waitFor();
  await audit("care plan desktop");
  await page.getByRole("button",{name:/Confirm taken: Prescribed/}).click();await page.locator("#dose-list .care-badge.taken").waitFor();
  assert.equal(await page.locator("#care-history tbody tr").count(),1);
  await page.locator("#a11y-size").selectOption("extra");await page.locator("#a11y-contrast").check();
  await page.setViewportSize({width:390,height:844});await audit("care mobile large text high contrast");
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await page.locator("#care-language").selectOption("hi");assert.equal(await page.locator("html").getAttribute("lang"),"hi");await page.getByRole("heading",{name:"आज की दवा लेने की पुष्टि",exact:true}).waitFor();
  await page.locator("#care-language").selectOption("mr");await page.getByRole("heading",{name:"आजच्या औषधांची पुष्टी",exact:true}).waitFor();await audit("care Marathi mobile");
  await page.locator("#care-language").selectOption("en");await page.locator("#a11y-size").selectOption("standard");await page.locator("#a11y-contrast").uncheck();
  await page.setViewportSize({width:320,height:900});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await page.goto(base+"/activity",{waitUntil:"networkidle"});await audit("activity 320px");
  await page.setViewportSize({width:1280,height:900});await audit("activity desktop");assert.deepEqual(errors,[]);
  console.log(JSON.stringify({views:audits.length,axeViolations:0,javascriptErrors:0,keyboard:"passed",confirmation:"passed",languages:"passed",mobileReflow:"passed"}));
} finally {
  if(browser)await browser.close();
  server.kill();
  await Promise.race([new Promise(resolve=>server.once("exit",resolve)),new Promise(resolve=>setTimeout(resolve,3000))]);
  const actual=await fs.realpath(temporary);const parent=await fs.realpath(os.tmpdir());
  assert(path.relative(parent,actual)&&!path.relative(parent,actual).startsWith(".."));
  await fs.rm(temporary,{recursive:true,force:true});
}
