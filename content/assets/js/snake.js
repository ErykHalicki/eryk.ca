
const canvas = document.getElementById('background-canvas');
const context = canvas.getContext('2d');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const SEGMENT_SIZE = 35;

function snapToGrid(val) {
  return Math.floor(val / SEGMENT_SIZE) * SEGMENT_SIZE;
}

const snake = {
  segments: [],
  direction: 'left',
  grow: 0
};

const startX = snapToGrid(canvas.width - 250);
const startY = snapToGrid(300);
for (let i = 0; i < 10; i++) {
  snake.segments.push({ x: startX, y: startY + i * SEGMENT_SIZE });
}

const UPDATE_MS = 100;
let pellets = [];

function spawnRandomPellet() {
  const cols = Math.floor(canvas.width / SEGMENT_SIZE);
  const rows = Math.floor(canvas.height / SEGMENT_SIZE);
  pellets.push({
    x: Math.floor(Math.random() * cols) * SEGMENT_SIZE,
    y: Math.floor(Math.random() * rows) * SEGMENT_SIZE
  });
}

spawnRandomPellet();

window.addEventListener('click', (e) => {
  pellets.push({ x: snapToGrid(e.clientX), y: snapToGrid(e.clientY) });
});

function getOppositeDirection(dir) {
  if (dir === 'up') return 'down';
  if (dir === 'down') return 'up';
  if (dir === 'left') return 'right';
  if (dir === 'right') return 'left';
  return dir;
}

function getRandomDirection() {
  const all = ['up', 'down', 'left', 'right'];
  return all.filter(d => d !== getOppositeDirection(snake.direction))[Math.floor(Math.random() * 3)];
}

function tick() {
  // 1. Decide direction.
  if (pellets.length > 0) {
    const head = snake.segments[0];

    // Pick nearest pellet by Manhattan distance.
    let target = pellets[0];
    let bestDist = Math.abs(target.x - head.x) + Math.abs(target.y - head.y);
    for (let i = 1; i < pellets.length; i++) {
      const d = Math.abs(pellets[i].x - head.x) + Math.abs(pellets[i].y - head.y);
      if (d < bestDist) { bestDist = d; target = pellets[i]; }
    }

    const dx = target.x - head.x;
    const dy = target.y - head.y;
    const absDx = Math.abs(dx);
    const absDy = Math.abs(dy);

    // Move along shorter axis first; once aligned, switch to the other.
    // Prefer the axis with smaller distance, falling back to the other if already aligned.
    let preferred = null;
    let fallback = null;

    if (absDx <= absDy) {
      preferred = dx > 0 ? 'right' : dx < 0 ? 'left' : null;
      fallback  = dy > 0 ? 'down'  : dy < 0 ? 'up'   : null;
    } else {
      preferred = dy > 0 ? 'down'  : dy < 0 ? 'up'   : null;
      fallback  = dx > 0 ? 'right' : dx < 0 ? 'left' : null;
    }

    // Use preferred unless it would reverse us; fall back to other axis.
    const opp = getOppositeDirection(snake.direction);
    if (preferred && preferred !== opp) {
      snake.direction = preferred;
    } else if (fallback && fallback !== opp) {
      snake.direction = fallback;
    }
  } else {
    snake.direction = getRandomDirection();
  }

  // 2. Move.
  const head = { ...snake.segments[0] };
  if (snake.direction === 'up') {
    head.y -= SEGMENT_SIZE;
    if (head.y < 0) head.y = canvas.height - SEGMENT_SIZE;
  } else if (snake.direction === 'down') {
    head.y += SEGMENT_SIZE;
    if (head.y >= canvas.height) head.y = 0;
  } else if (snake.direction === 'left') {
    head.x -= SEGMENT_SIZE;
    if (head.x < 0) head.x = canvas.width - SEGMENT_SIZE;
  } else if (snake.direction === 'right') {
    head.x += SEGMENT_SIZE;
    if (head.x >= canvas.width) head.x = 0;
  }
  snake.segments.unshift(head);

  const eaten = pellets.findIndex(p => p.x === head.x && p.y === head.y);
  if (eaten !== -1) {
    pellets.splice(eaten, 1);
    snake.grow += 1;
    if (pellets.length === 0) spawnRandomPellet();
  }

  if (snake.grow > 0) {
    snake.grow--;
  } else {
    snake.segments.pop();
  }

  if (snake.segments.length >= 50) {
    snake.segments.length = 10;
    snake.grow = 0;
  }

  // 3. Draw.
  context.clearRect(0, 0, canvas.width, canvas.height);

  pellets.forEach(p => {
    context.fillStyle = 'rgb(220, 50, 50)';
    context.fillRect(p.x, p.y, SEGMENT_SIZE, SEGMENT_SIZE);

    // Pixelated leaf sitting on top of pellet.
    const ps = 4;
    const leafPixels = [
      [0,0],[1,0],
      [0,1],[1,1],
      [1,2],[2,2],
      [1,3],[2,3]
    ];
    const leafX = p.x + Math.floor(SEGMENT_SIZE * 0.1);
    const leafY = p.y - 2 * ps;
    context.fillStyle = 'rgb(60, 180, 60)';
    for (const [c, r] of leafPixels) {
      context.fillRect(leafX + c * ps, leafY + r * ps, ps, ps);
    }
  });

  // Draw body first, then head on top.
  for (let i = snake.segments.length - 1; i >= 1; i--) {
    const seg = snake.segments[i];
    context.fillStyle = 'rgb(230, 139, 23)';
    context.fillRect(seg.x, seg.y, SEGMENT_SIZE, SEGMENT_SIZE);
  }

  const headSeg = snake.segments[0];
  context.fillStyle = 'rgb(230, 139, 23)';
  context.fillRect(headSeg.x, headSeg.y, SEGMENT_SIZE, SEGMENT_SIZE);
  const eyeSize   = SEGMENT_SIZE / 8;
  const eyeOffset = SEGMENT_SIZE / 4;
  context.fillStyle = 'black';
  context.beginPath();
  context.arc(headSeg.x + eyeOffset, headSeg.y + eyeOffset, eyeSize, 0, Math.PI * 2);
  context.fill();
  context.beginPath();
  context.arc(headSeg.x + SEGMENT_SIZE - eyeOffset, headSeg.y + eyeOffset, eyeSize, 0, Math.PI * 2);
  context.fill();
}

setInterval(tick, UPDATE_MS);
