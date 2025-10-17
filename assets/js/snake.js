

const canvas = document.getElementById('background-canvas');
const context = canvas.getContext('2d');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const snake = {
  segments: [],
  direction: 'left',
  segmentSize: 35
};

for (let i = 0; i < 10; i++) {
  snake.segments.push({ x: canvas.width-250 , y: 300 + i * snake.segmentSize });
}

function getRandomDirection() {
  const directions = ['up', 'down', 'left', 'right'];
  
  // Filter out the opposite direction
  const filteredDirections = directions.filter(dir => dir !== getOppositeDirection(snake.direction));
  
  // Pick a random direction from the filtered list
  return filteredDirections[Math.floor(Math.random() * filteredDirections.length)];
}

function getOppositeDirection(dir) {
  if (dir === 'up') return 'down';
  if (dir === 'down') return 'up';
  if (dir === 'left') return 'right';
  if (dir === 'right') return 'left';
  return dir;
}
function moveSnake() {
  const head = { ...snake.segments[0] };

  if (snake.direction === 'up') {
    head.y -= snake.segmentSize;
    if (head.y < 0) {
      head.y = canvas.height - snake.segmentSize;
    }
  } else if (snake.direction === 'down') {
    head.y += snake.segmentSize;
    if (head.y >= canvas.height) {
      head.y = 0;
    }
  } else if (snake.direction === 'left') {
    head.x -= snake.segmentSize;
    if (head.x < 0) {
      head.x = canvas.width - snake.segmentSize;
    }
  } else if (snake.direction === 'right') {
    head.x += snake.segmentSize;
    if (head.x >= canvas.width) {
      head.x = 0;
    }
  }

  snake.segments.unshift(head);
  snake.segments.pop();
}
function drawSnake() {
  context.clearRect(0, 0, canvas.width, canvas.height);

  snake.segments.forEach((segment, index) => {
    context.fillStyle = 'rgb(230, 139, 23)';
    context.fillRect(segment.x, segment.y, snake.segmentSize, snake.segmentSize);
    
    // Add eyes to the head
    if (index === 0) {
      const eyeSize = snake.segmentSize / 8;
      const eyeOffset = snake.segmentSize / 4;
      context.fillStyle = 'black';
      context.beginPath();
      context.arc(segment.x + eyeOffset, segment.y + eyeOffset, eyeSize, 0, Math.PI * 2);
      context.fill();
      context.beginPath();
      context.arc(segment.x + snake.segmentSize - eyeOffset, segment.y + eyeOffset, eyeSize, 0, Math.PI * 2);
      context.fill();
    }
  });
}
function changeDirection() {
  snake.direction = getRandomDirection();
}

setInterval(changeDirection, 1500); // Change direction every 2 seconds
setInterval(moveSnake, 150); // Update the snake's position every 200 milliseconds (5 times a second)

setInterval(drawSnake, 150); // Redraw the snake at the same rate

