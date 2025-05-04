function showEnlargedImage(img) {
    const overlay = document.getElementById('imageOverlay');
    const enlargedImg = document.getElementById('enlargedImage');
    
    enlargedImg.src = img.src;
    overlay.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeEnlargedImage() {
    const overlay = document.getElementById('imageOverlay');
    overlay.classList.remove('active');
    document.body.style.overflow = 'auto';
}

// Close overlay when clicking outside the image
document.getElementById('imageOverlay').addEventListener('click', function(e) {
    if (e.target === this) {
        closeEnlargedImage();
    }
});

// Close on escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeEnlargedImage();
    }
});