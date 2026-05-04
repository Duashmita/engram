// dino.js — mini infinite-runner for the loading screen.
// Matches the app dark theme. Space / ArrowUp / tap to jump.

export class DinoGame {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    this.raf    = null;
    this.running = false;

    this._onKey   = e => {
      if (e.code !== 'Space' && e.code !== 'ArrowUp') return;
      const tag = document.activeElement?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      e.preventDefault();
      this._jump();
    };
    this._onClick = () => this._jump();
    document.addEventListener('keydown', this._onKey);
    canvas.addEventListener('click', this._onClick);

    this._reset();
  }

  _reset() {
    const H = this.canvas.height;
    this.ground    = H - 44;
    this.dino      = { x: 72, y: 0, vy: 0, w: 28, h: 38, grounded: true };
    this.dino.y    = this.ground - this.dino.h;
    this.obstacles = [];
    this.speed     = 4;
    this.frame     = 0;
    this.score     = 0;
    this.dead      = false;
  }

  start() {
    this._reset();
    this.running = true;
    this._tick();
  }

  stop() {
    this.running = false;
    if (this.raf) { cancelAnimationFrame(this.raf); this.raf = null; }
  }

  destroy() {
    this.stop();
    document.removeEventListener('keydown', this._onKey);
    this.canvas.removeEventListener('click', this._onClick);
  }

  _jump() {
    if (!this.running) return;
    if (this.dead) { this._reset(); return; }
    if (this.dino.grounded) { this.dino.vy = -13; this.dino.grounded = false; }
  }

  _tick() {
    if (!this.running) return;
    this._update();
    this._draw();
    this.raf = requestAnimationFrame(() => this._tick());
  }

  _update() {
    if (this.dead) return;
    const d = this.dino;

    d.vy += 0.65;
    d.y  += d.vy;
    if (d.y >= this.ground - d.h) {
      d.y = this.ground - d.h;
      d.vy = 0;
      d.grounded = true;
    }

    this.frame++;
    const gap = Math.max(55, 90 - Math.floor(this.score / 80));
    if (this.frame % gap === 0) {
      const h = 22 + Math.random() * 28;
      const w = 14 + Math.random() * 12;
      this.obstacles.push({ x: this.canvas.width + 10, y: this.ground - h, w, h });
    }
    this.obstacles = this.obstacles
      .map(o => ({ ...o, x: o.x - this.speed }))
      .filter(o => o.x > -60);

    this.speed = 4 + this.score / 400;
    this.score++;

    for (const o of this.obstacles) {
      const pad = 5;
      if (
        d.x + d.w - pad > o.x + pad &&
        d.x + pad < o.x + o.w - pad &&
        d.y + d.h - pad > o.y + pad
      ) { this.dead = true; }
    }
  }

  _draw() {
    const ctx = this.ctx, W = this.canvas.width, H = this.canvas.height;
    const d = this.dino;

    ctx.clearRect(0, 0, W, H);

    // ground
    ctx.strokeStyle = '#2d3744';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(0, this.ground); ctx.lineTo(W, this.ground); ctx.stroke();

    // dino body
    ctx.fillStyle = this.dead ? '#ff5d6c' : '#6aa3ff';
    ctx.beginPath();
    ctx.roundRect(d.x, d.y, d.w, d.h, 4);
    ctx.fill();

    // dino eye
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(d.x + d.w - 9, d.y + 7, 5, 5);

    // dino legs (animate)
    if (d.grounded) {
      const legPhase = Math.floor(this.frame / 6) % 2;
      ctx.fillStyle = '#4a7acc';
      ctx.fillRect(d.x + 4,       d.y + d.h,     8, legPhase === 0 ? 6 : 10);
      ctx.fillRect(d.x + d.w - 12, d.y + d.h,    8, legPhase === 0 ? 10 : 6);
    }

    // obstacles (cacti — segmented green)
    for (const o of this.obstacles) {
      ctx.fillStyle = '#4ddb8a';
      ctx.beginPath();
      ctx.roundRect(o.x, o.y, o.w, o.h, 3);
      ctx.fill();
      // arm
      ctx.fillRect(o.x - 6, o.y + o.h * 0.3, 6, 8);
      ctx.fillRect(o.x + o.w, o.y + o.h * 0.5, 6, 8);
    }

    // score
    ctx.fillStyle = '#3d4f63';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(String(Math.floor(this.score / 10)).padStart(5, '0'), W - 16, 22);

    // hint / dead text
    ctx.textAlign = 'center';
    if (this.dead) {
      ctx.fillStyle = '#d6deeb';
      ctx.font = '12px "JetBrains Mono", monospace';
      ctx.fillText('GAME OVER  ·  space to restart', W / 2, H / 2 - 4);
    } else if (this.score < 80) {
      ctx.fillStyle = '#3d4f63';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText('space / tap to jump', W / 2, H / 2);
    }
  }
}
