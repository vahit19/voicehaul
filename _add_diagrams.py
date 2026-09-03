"""Adds two block diagrams and two charts to the static report. Run once."""
import io

# ---------------------------------------------------------------------------
# Diagram 1: the measurement loop, and the asymmetry that is the whole finding
# ---------------------------------------------------------------------------

LOOP = """
  <figure class="fig">
    <svg viewBox="0 0 900 300" role="img" xmlns="http://www.w3.org/2000/svg"
      aria-label="One turn: the caller's state becomes an utterance, the model
      replies, the reply is measured into a delivery vector, and that vector is
      scored twice - perceived empathy, which is what a rater sees and which
      feeds nothing, and calibration, which drives the caller's next state.">
      <defs>
        <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
          markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>
        </marker>
        <marker id="ar-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
          markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="var(--accent)"/>
        </marker>
        <marker id="ar-x" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
          markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="var(--amber)"/>
        </marker>
      </defs>

      <rect x="14" y="112" width="132" height="52" rx="3" fill="none"
        stroke="currentColor" stroke-width="1.2"/>
      <text x="80" y="134" text-anchor="middle" class="dlab">caller state</text>
      <text x="80" y="151" text-anchor="middle" class="dsub">6-dim affect</text>

      <line x1="146" y1="138" x2="212" y2="138" stroke="currentColor"
        stroke-width="1.2" marker-end="url(#ar)"/>
      <text x="179" y="128" text-anchor="middle" class="dedge">renders</text>

      <rect x="212" y="112" width="128" height="52" rx="3" fill="none"
        stroke="currentColor" stroke-width="1.2"/>
      <text x="276" y="134" text-anchor="middle" class="dlab">utterance</text>
      <text x="276" y="151" text-anchor="middle" class="dsub">what is said</text>

      <line x1="340" y1="138" x2="404" y2="138" stroke="currentColor"
        stroke-width="1.2" marker-end="url(#ar)"/>

      <rect x="404" y="106" width="126" height="64" rx="3" fill="var(--accent-soft)"
        stroke="var(--accent)" stroke-width="1.4"/>
      <text x="467" y="132" text-anchor="middle" class="dlab">model</text>
      <text x="467" y="149" text-anchor="middle" class="dsub">under test</text>

      <line x1="530" y1="138" x2="594" y2="138" stroke="currentColor"
        stroke-width="1.2" marker-end="url(#ar)"/>
      <text x="562" y="128" text-anchor="middle" class="dedge">reply</text>

      <rect x="594" y="106" width="140" height="64" rx="3" fill="none"
        stroke="currentColor" stroke-width="1.2"/>
      <text x="664" y="130" text-anchor="middle" class="dlab">delivery vector</text>
      <text x="664" y="147" text-anchor="middle" class="dsub">rate, warmth, length,</text>
      <text x="664" y="161" text-anchor="middle" class="dsub">apology, acknowledgement</text>

      <path d="M734 122 L774 122 L774 52 L810 52" fill="none" stroke="var(--amber)"
        stroke-width="1.4" marker-end="url(#ar-x)"/>
      <path d="M734 154 L774 154 L774 232 L810 232" fill="none" stroke="var(--accent)"
        stroke-width="1.4" marker-end="url(#ar-a)"/>

      <rect x="810" y="28" width="78" height="48" rx="3" fill="none"
        stroke="var(--amber)" stroke-width="1.3"/>
      <text x="849" y="48" text-anchor="middle" class="dlab" fill="var(--amber)">perceived</text>
      <text x="849" y="64" text-anchor="middle" class="dlab" fill="var(--amber)">empathy</text>

      <rect x="810" y="208" width="78" height="48" rx="3" fill="none"
        stroke="var(--accent)" stroke-width="1.3"/>
      <text x="849" y="228" text-anchor="middle" class="dlab" fill="var(--accent)">calibration</text>
      <text x="849" y="244" text-anchor="middle" class="dsub" fill="var(--accent)">hidden</text>

      <text x="880" y="94" text-anchor="end" class="dnote" fill="var(--amber)">
        what a rater scores</text>
      <text x="880" y="274" text-anchor="end" class="dnote" fill="var(--accent)">
        what moves the caller</text>

      <path d="M810 232 L768 232 L768 268 L80 268 L80 164" fill="none"
        stroke="var(--accent)" stroke-width="1.4" stroke-dasharray="5 4"
        marker-end="url(#ar-a)"/>
      <text x="424" y="262" text-anchor="middle" class="dedge" fill="var(--accent)">
        drives the next state</text>

      <line x1="849" y1="76" x2="849" y2="196" stroke="var(--amber)"
        stroke-width="1.2" stroke-dasharray="3 5"/>
      <text x="862" y="140" class="dnote" fill="var(--amber)">feeds</text>
      <text x="862" y="155" class="dnote" fill="var(--amber)">nothing</text>
    </svg>
    <figcaption>One turn. The same delivery vector is scored twice, and only one
    of the two scores changes what happens next. The observable score is the one
    with no arrow leaving it.</figcaption>
  </figure>
"""

# ---------------------------------------------------------------------------
# Diagram 2: three raters, and the arrow real data does not have
# ---------------------------------------------------------------------------

RATERS = """
  <figure class="fig">
    <svg viewBox="0 0 900 330" role="img" xmlns="http://www.w3.org/2000/svg"
      aria-label="One turn is scored by a human panel and by an LLM judge. The
      customer's estimate comes from correlating those two and dividing out the
      panel's own unreliability. A third path, the latent truth, exists only in
      simulation and is what validates the estimate.">
      <defs>
        <marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
          markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="currentColor"/>
        </marker>
        <marker id="ar2-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
          markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 L10 5 L0 10 z" fill="var(--accent)"/>
        </marker>
      </defs>

      <rect x="14" y="132" width="118" height="52" rx="3" fill="none"
        stroke="currentColor" stroke-width="1.2"/>
      <text x="73" y="154" text-anchor="middle" class="dlab">one turn</text>
      <text x="73" y="171" text-anchor="middle" class="dsub">caller + reply</text>

      <path d="M132 148 L200 78" fill="none" stroke="currentColor" stroke-width="1.2"
        marker-end="url(#ar2)"/>
      <path d="M132 158 L200 158" fill="none" stroke="currentColor" stroke-width="1.2"
        marker-end="url(#ar2)"/>
      <path d="M132 168 L200 254" fill="none" stroke="var(--accent)" stroke-width="1.3"
        stroke-dasharray="5 4" marker-end="url(#ar2-a)"/>

      <rect x="200" y="52" width="150" height="52" rx="3" fill="none"
        stroke="currentColor" stroke-width="1.2"/>
      <text x="275" y="74" text-anchor="middle" class="dlab">human panel</text>
      <text x="275" y="91" text-anchor="middle" class="dsub">k raters, noisy</text>

      <rect x="200" y="132" width="150" height="52" rx="3" fill="none"
        stroke="currentColor" stroke-width="1.2"/>
      <text x="275" y="154" text-anchor="middle" class="dlab">LLM judge</text>
      <text x="275" y="171" text-anchor="middle" class="dsub">cheap, unproven</text>

      <rect x="200" y="228" width="150" height="52" rx="3" fill="none"
        stroke="var(--accent)" stroke-width="1.4" stroke-dasharray="5 4"/>
      <text x="275" y="250" text-anchor="middle" class="dlab" fill="var(--accent)">latent truth</text>
      <text x="275" y="267" text-anchor="middle" class="dsub" fill="var(--accent)">simulation only</text>

      <path d="M350 78 L430 118" fill="none" stroke="currentColor" stroke-width="1.2"
        marker-end="url(#ar2)"/>
      <path d="M350 158 L430 138" fill="none" stroke="currentColor" stroke-width="1.2"
        marker-end="url(#ar2)"/>

      <rect x="430" y="102" width="176" height="58" rx="3" fill="none"
        stroke="currentColor" stroke-width="1.2"/>
      <text x="518" y="124" text-anchor="middle" class="dlab">correlate, then</text>
      <text x="518" y="141" text-anchor="middle" class="dlab">disattenuate</text>
      <text x="518" y="155" text-anchor="middle" class="dsub">divide out panel noise</text>

      <line x1="606" y1="131" x2="672" y2="131" stroke="currentColor"
        stroke-width="1.2" marker-end="url(#ar2)"/>

      <rect x="672" y="102" width="150" height="58" rx="3" fill="none"
        stroke="currentColor" stroke-width="1.4"/>
      <text x="747" y="126" text-anchor="middle" class="dlab">estimated rho</text>
      <text x="747" y="145" text-anchor="middle" class="dsub">what a customer gets</text>

      <path d="M350 254 L660 254 L660 176" fill="none" stroke="var(--accent)"
        stroke-width="1.4" stroke-dasharray="5 4"/>
      <path d="M660 176 L700 176 L700 164" fill="none" stroke="var(--accent)"
        stroke-width="1.4" marker-end="url(#ar2-a)"/>
      <text x="505" y="246" text-anchor="middle" class="dedge" fill="var(--accent)">
        is the estimate right?</text>

      <text x="886" y="200" text-anchor="end" class="dnote" fill="var(--accent)">
        this arrow does not exist on real data:</text>
      <text x="886" y="216" text-anchor="end" class="dnote" fill="var(--accent)">
        both measurements carry error and</text>
      <text x="886" y="232" text-anchor="end" class="dnote" fill="var(--accent)">
        neither one is the reference</text>
    </svg>
    <figcaption>The estimator a customer can run uses only the two solid paths.
    The dashed path is the validation, and it is the reason this is built in a
    simulator before it is pointed at a contract.</figcaption>
  </figure>
"""

CSS = """.fig{margin:26px 0;max-width:100%}
.fig svg{max-width:100%;height:auto;color:var(--ink-2)}
.fig figcaption{font-size:13.5px;color:var(--muted);margin-top:10px;
  max-width:62ch;line-height:1.5}
.dlab{font-family:"IBM Plex Sans",sans-serif;font-size:13px;font-weight:500;
  fill:var(--ink)}
.dsub{font-family:"IBM Plex Mono",monospace;font-size:10px;fill:var(--faint)}
.dedge{font-family:"IBM Plex Mono",monospace;font-size:10.5px;fill:var(--muted)}
.dnote{font-family:"IBM Plex Mono",monospace;font-size:10.5px;fill:var(--muted)}
"""

CHARTS = """
  <div class="col">
    <h3>What the correlation looks like</h3>
    <p class="note">Each dot is one turn. The horizontal axis is the quality the
    turn actually had; the vertical axis is what each rater said about it. A
    rater that tracks quality produces a diagonal cloud; one that does not
    produces a horizontal band.</p>
  </div>
  <div class="panel">
    <div class="scroll"><svg id="sub-scatter" viewBox="0 0 940 380" role="img"
      aria-label="Human panel scores rise with true quality; judge scores do not"></svg></div>
    <div class="legend">
      <span><i style="background:var(--accent)"></i>human panel (mean of 3)</span>
      <span><i style="background:var(--alarm)"></i>LLM judge</span>
    </div>
  </div>

  <div class="col">
    <h3>And how far the answer moves between segments</h3>
  </div>
  <div class="panel">
    <div class="scroll"><svg id="sub-bars" viewBox="0 0 940 300" role="img"
      aria-label="Substitution ratio by caller segment for both dimensions"></svg></div>
  </div>
"""

JS = """
/* ---------- judge substitution: scatter and segment bars ---------- */
(function(){
  var S = D.substitution;
  if(!S) return;
  var PL = {distressed_billing:"billing", hostile_escalation:"hostile",
            confused_elderly:"confused", grieving_claim:"grieving",
            cautious_optimist:"calm"};

  /* scatter: two panels, panel-vs-truth and judge-vs-truth overlaid */
  (function(){
    var W=940,H=380,pad={l:52,r:20,t:34,b:52},gap=48;
    var dims=["perceived_empathy","actual_help"];
    var titles={perceived_empathy:"perceived empathy",
                actual_help:"did it actually help"};
    var panelW=(W-pad.l-pad.r-gap)/2, s="";
    dims.forEach(function(dim,di){
      var x0=pad.l+di*(panelW+gap);
      var pts=S.pairs[dim]||[];
      var px=function(t){ return x0+panelW*Math.max(0,Math.min(1,t)); };
      var py=function(v){ return pad.t+(H-pad.t-pad.b)*(1-(v-1)/6); };
      var i,y;
      for(i=0;i<=3;i++){
        y=pad.t+(H-pad.t-pad.b)*i/3;
        s+='<line class="gridline" x1="'+x0+'" y1="'+y.toFixed(1)+'" x2="'+
           (x0+panelW)+'" y2="'+y.toFixed(1)+'"/>';
        if(di===0){ s+='<text class="axis" x="'+(x0-8)+'" y="'+(y+3.5).toFixed(1)+
          '" text-anchor="end">'+(7-2*i)+'</text>'; }
      }
      pts.forEach(function(p){
        s+='<circle cx="'+px(p.theta).toFixed(1)+'" cy="'+py(p.panel).toFixed(1)+
           '" r="3" fill="var(--accent)" opacity=".45"/>';
      });
      pts.forEach(function(p){
        s+='<circle cx="'+px(p.theta).toFixed(1)+'" cy="'+py(p.judge).toFixed(1)+
           '" r="3" fill="var(--alarm)" opacity=".55"/>';
      });
      /* least-squares line for each series, so the trend is not eyeballed */
      [["panel","var(--accent)"],["judge","var(--alarm)"]].forEach(function(kv){
        var k=kv[0], n=pts.length; if(n<3) return;
        var mx=0,my=0,i2; for(i2=0;i2<n;i2++){mx+=pts[i2].theta;my+=pts[i2][k];}
        mx/=n; my/=n;
        var num=0,den=0;
        for(i2=0;i2<n;i2++){num+=(pts[i2].theta-mx)*(pts[i2][k]-my);
                            den+=(pts[i2].theta-mx)*(pts[i2].theta-mx);}
        if(den<=0) return;
        var b=num/den, a=my-b*mx;
        s+='<line x1="'+px(0)+'" y1="'+py(a).toFixed(1)+'" x2="'+px(1)+'" y2="'+
           py(a+b).toFixed(1)+'" stroke="'+kv[1]+'" stroke-width="2.2"/>';
      });
      s+='<text class="axlabel" x="'+(x0+panelW/2)+'" y="'+(pad.t-14)+
         '" text-anchor="middle">'+titles[dim]+'</text>';
      s+='<text class="axis" x="'+(x0+panelW/2)+'" y="'+(H-pad.b+30)+
         '" text-anchor="middle">true quality of the turn &rarr;</text>';
    });
    s+='<text class="axlabel" x="14" y="'+(pad.t-14)+'">rating 1-7</text>';
    document.getElementById("sub-scatter").innerHTML=s;
  })();

  /* segment bars: substitution ratio, log-ish scale capped at 2 */
  (function(){
    var W=940,H=300,pad={l:120,r:130,t:36,b:40};
    var dims=["perceived_empathy","actual_help"];
    var titles={perceived_empathy:"perceived empathy",
                actual_help:"did it actually help"};
    var rows=S.rows.filter(function(r){return r.segment!=="all";});
    var segs=[];
    rows.forEach(function(r){ if(segs.indexOf(r.segment)<0) segs.push(r.segment); });
    var band=(H-pad.t-pad.b)/segs.length, maxv=2.0, s="";
    var px=function(v){ return pad.l+(W-pad.l-pad.r)*Math.min(v,maxv)/maxv; };
    var t;
    for(t=0;t<=4;t++){
      var v=maxv*t/4, x=px(v);
      s+='<line class="gridline" x1="'+x.toFixed(1)+'" y1="'+pad.t+'" x2="'+
         x.toFixed(1)+'" y2="'+(H-pad.b)+'"/>';
      s+='<text class="axis" x="'+x.toFixed(1)+'" y="'+(H-pad.b+16)+
         '" text-anchor="middle">'+v.toFixed(1)+'</text>';
    }
    s+='<line x1="'+px(1)+'" y1="'+pad.t+'" x2="'+px(1)+'" y2="'+(H-pad.b)+
       '" stroke="var(--ink-2)" stroke-width="1.2" stroke-dasharray="4 4"/>';
    s+='<text class="axis" x="'+(px(1)+7)+'" y="'+(pad.t-8)+
       '" fill="var(--ink-2)">1 judge = 1 human</text>';
    segs.forEach(function(seg,i){
      var y0=pad.t+i*band;
      s+='<text class="axis" x="'+(pad.l-10)+'" y="'+(y0+band/2+4).toFixed(1)+
         '" text-anchor="end" fill="var(--ink-2)" style="font-size:12px">'+
         (PL[seg]||seg)+'</text>';
      dims.forEach(function(dim,di){
        var r=rows.filter(function(x){return x.segment===seg&&x.dimension===dim;})[0];
        if(!r) return;
        var h=band*0.32, y=y0+band*0.16+di*(h+3);
        var col=di===0?"var(--amber)":"var(--alarm)";
        var w=Math.max(1.5,px(r.ratio_estimated)-pad.l);
        s+='<rect x="'+pad.l+'" y="'+y.toFixed(1)+'" width="'+w.toFixed(1)+
           '" height="'+h.toFixed(1)+'" fill="'+col+'" rx="1"/>';
        s+='<text class="axis" x="'+(pad.l+w+7).toFixed(1)+'" y="'+
           (y+h/2+3.5).toFixed(1)+'" fill="'+col+'">'+
           r.ratio_estimated.toFixed(2)+'</text>';
      });
    });
    s+='<text class="axlabel" x="'+pad.l+'" y="'+(H-6)+
       '">human ratings one judge rating is worth</text>';
    s+='<text class="axis" x="'+(W-pad.r+14)+'" y="'+(pad.t+10)+
       '" fill="var(--amber)">'+titles.perceived_empathy+'</text>';
    s+='<text class="axis" x="'+(W-pad.r+14)+'" y="'+(pad.t+26)+
       '" fill="var(--alarm)">'+titles.actual_help+'</text>';
    document.getElementById("sub-bars").innerHTML=s;
  })();
})();
"""

p = "_web_template.html"
s = io.open(p, encoding="utf-8").read()
if ".fig{" in s:
    print("already present")
else:
    s = s.replace("footer{margin-top:56px;", CSS + "footer{margin-top:56px;")

    # loop diagram: right after the headline finding
    a1 = ('  <div class="col">\n    <div class="callout">\n      <p style="margin:0">'
          'The turn panel says the best model is')
    assert a1 in s, "headline callout not found"
    s = s.replace('<div class="ruler"></div>\n\n<section>\n  <div class="col">\n'
                  '    <p class="eyebrow">Explore</p>',
                  LOOP + '\n<div class="ruler"></div>\n\n<section>\n  <div class="col">\n'
                  '    <p class="eyebrow">Explore</p>', 1)

    # raters diagram + charts inside the substitution section
    a2 = '    <h3>The method</h3>'
    assert a2 in s
    s = s.replace(a2, '  </div>\n' + RATERS + CHARTS + '  <div class="col">\n' + a2, 1)

    s = s.replace("/* ---------- power calculator ---------- */",
                  JS + "\n/* ---------- power calculator ---------- */")
    io.open(p, "w", encoding="utf-8").write(s)
    print("2 diyagram + 2 grafik eklendi")
