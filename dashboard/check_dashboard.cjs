const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(require('path').join(__dirname,'../resultados/dashboard/index.html'),'utf8');
const payload=html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const elements={};
const element=id=>elements[id]??=( {textContent:id==='data'?payload:'',innerHTML:'',value:'',addEventListener(){}} );
const context=vm.createContext({Intl,document:{getElementById:element,querySelectorAll:()=>[]}});
vm.runInContext(scripts[0][1],context);
const data=JSON.parse(payload);
assert.equal(data.models.length,3);assert.equal(data.cv.length,24);
for(let active=0;active<4;active++)for(const focus of ['all','anomalies','errors']){
 vm.runInContext(`active=${active};focus='${focus}';render()`,context);
 assert(!elements.view.innerHTML.includes('NaN'));
 assert(elements.view.innerHTML.includes('<svg'));
 assert.equal((elements.view.innerHTML.match(/<svg/g)||[]).length,active===3?3:1);
 for(let i=0;i<3;i++){
  const flows=vm.runInContext(`flowValues(data.models[${i}])`,context);
  const m=data.models[i],expected=focus==='all'?2000:focus==='anomalies'?20:m.fp+m.fn;
  assert.equal(flows.reduce((s,f)=>s+f.n,0),expected);
  const chart=vm.runInContext(`sankey(data.models[${i}])`,context);
  assert.equal((chart.match(/<path /g)||[]).length,flows.filter(f=>f.n>0).length);
 }
}
console.log('OK: 4 visões × 3 filtros; fluxos, totais e execução JavaScript verificados.');
