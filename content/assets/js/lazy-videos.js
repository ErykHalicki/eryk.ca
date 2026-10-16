const videoObserver = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (entry.isIntersecting) {
      entry.target.play().catch(() => {});
    } else {
      entry.target.pause();
    }
  }
}, { rootMargin: '100px 0px' });

document.querySelectorAll('video.timeline-thumb').forEach((video) => videoObserver.observe(video));
