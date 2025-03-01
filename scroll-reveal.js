document.addEventListener('DOMContentLoaded', () => {
    // Initialize scroll progress indicator
    const scrollProgress = document.querySelector('.scroll-progress');
    
    // Initialize all sections for reveal animations
    initRevealSections();
    
    // Initialize parallax elements
    initParallaxElements();
    
    // Update scroll progress and check for reveals on scroll
    window.addEventListener('scroll', () => {
        // Update scroll progress bar
        updateScrollProgress();
        
        // Check for elements to reveal
        checkReveal();
        
        // Update parallax elements
        updateParallaxElements();
    });
    
    // Initial check for elements in viewport on page load
    setTimeout(() => {
        checkReveal();
        updateScrollProgress();
    }, 300); // Small delay for initial animations
});

// Update the scroll progress indicator
function updateScrollProgress() {
    const scrollProgress = document.querySelector('.scroll-progress');
    if (!scrollProgress) return;
    
    const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
    const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrollPercent = (scrollTop / scrollHeight) * 100;
    
    scrollProgress.style.width = `${scrollPercent}%`;
    
    // Add glitch effect at certain scroll percentages
    if (scrollPercent > 25 && scrollPercent < 26 || 
        scrollPercent > 50 && scrollPercent < 51 || 
        scrollPercent > 75 && scrollPercent < 76) {
        scrollProgress.style.filter = 'hue-rotate(90deg) brightness(1.5)';
        setTimeout(() => {
            scrollProgress.style.filter = 'none';
        }, 400); // Longer glitch effect
    }
}

// Initialize all sections with reveal classes
function initRevealSections() {
    // Add reveal classes to main sections if not already present
    const sections = document.querySelectorAll('section');
    sections.forEach((section, index) => {
        if (!section.classList.contains('reveal-section') && 
            !section.classList.contains('reveal-left') && 
            !section.classList.contains('reveal-right') && 
            !section.classList.contains('reveal-scale') &&
            !section.classList.contains('cyber-reveal')) {
            
            // Alternate between different reveal effects for variety
            switch (index % 5) {
                case 0:
                    section.classList.add('reveal-section');
                    break;
                case 1:
                    section.classList.add('reveal-left');
                    break;
                case 2:
                    section.classList.add('reveal-right');
                    break;
                case 3:
                    section.classList.add('reveal-scale');
                    break;
                case 4:
                    section.classList.add('cyber-reveal');
                    break;
            }
        }
        
        // Add staggered reveal to child elements in certain sections
        if (section.id === 'services' || section.id === 'data-science' || section.id === 'contact') {
            const container = section.querySelector('.service-grid') || 
                             section.querySelector('.project-container') || 
                             section.querySelector('.contact-container');
            
            if (container && !container.classList.contains('reveal-stagger')) {
                container.classList.add('reveal-stagger');
                
                // Set reveal index for staggered animation with longer delays
                const children = container.children;
                Array.from(children).forEach((child, i) => {
                    child.style.setProperty('--reveal-index', i);
                });
            }
        }
    });
    
    // Add reveal classes to other important elements
    const cards = document.querySelectorAll('.service-card, .project-card, .data-project-card');
    cards.forEach((card, index) => {
        if (!card.classList.contains('cyber-reveal')) {
            card.classList.add('cyber-reveal');
        }
    });
}

// Check if elements should be revealed based on scroll position
function checkReveal() {
    const revealElements = document.querySelectorAll('.reveal-section, .reveal-left, .reveal-right, .reveal-scale, .cyber-reveal, .reveal-stagger, .sticky-content');
    
    revealElements.forEach(element => {
        const elementTop = element.getBoundingClientRect().top;
        const elementBottom = element.getBoundingClientRect().bottom;
        const windowHeight = window.innerHeight;
        
        // Reveal when element is just 5% into the viewport for a more gradual reveal
        const revealPoint = windowHeight * 0.05;
        
        if (elementTop < windowHeight - revealPoint && elementBottom > 0) {
            // Add a small random delay for more natural-looking reveals
            const randomDelay = Math.random() * 150;
            
            setTimeout(() => {
                element.classList.add('revealed');
                
                // Add glitch effect to cyber elements when revealed
                if (element.classList.contains('cyber-reveal')) {
                    addGlitchEffect(element);
                }
            }, randomDelay);
        } else if (elementBottom < -windowHeight || elementTop > windowHeight * 2) {
            // Optional: Hide elements again when they're far out of viewport
            // This creates a fresh reveal when scrolling back
            element.classList.remove('revealed');
            element.removeAttribute('data-glitched');
        }
    });
}

// Add temporary glitch effect to element
function addGlitchEffect(element) {
    // Only add effect if it hasn't been added before
    if (!element.dataset.glitched) {
        element.dataset.glitched = 'true';
        
        // Create glitch overlay
        const glitchOverlay = document.createElement('div');
        glitchOverlay.classList.add('glitch-overlay');
        glitchOverlay.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, var(--neon-cyan), var(--neon-pink));
            mix-blend-mode: overlay;
            opacity: 0;
            z-index: 1;
            pointer-events: none;
        `;
        
        element.style.position = element.style.position || 'relative';
        element.appendChild(glitchOverlay);
        
        // Animate glitch effect with longer intervals
        let glitchCount = 0;
        const glitchInterval = setInterval(() => {
            glitchOverlay.style.opacity = Math.random() * 0.5;
            glitchOverlay.style.clipPath = `polygon(
                ${Math.random() * 10}% ${Math.random() * 10}%, 
                ${90 + Math.random() * 10}% ${Math.random() * 10}%, 
                ${90 + Math.random() * 10}% ${90 + Math.random() * 10}%, 
                ${Math.random() * 10}% ${90 + Math.random() * 10}%
            )`;
            
            glitchCount++;
            if (glitchCount > 5) {
                clearInterval(glitchInterval);
                glitchOverlay.style.opacity = 0;
            }
        }, 180); // Slower glitch effect
    }
}

// Initialize parallax elements
function initParallaxElements() {
    const containers = document.querySelectorAll('.parallax-container');
    
    containers.forEach(container => {
        const children = container.children;
        Array.from(children).forEach(child => {
            if (!child.classList.contains('parallax-element')) {
                child.classList.add('parallax-element');
                
                // Set slower parallax speed
                const speed = (Math.random() * 0.2) + 0.05; // Reduced speed for slower effect
                child.setAttribute('data-parallax-speed', speed);
            }
        });
    });
}

// Update parallax elements based on scroll position
function updateParallaxElements() {
    const parallaxElements = document.querySelectorAll('.parallax-element');
    
    parallaxElements.forEach(element => {
        const speed = parseFloat(element.getAttribute('data-parallax-speed')) || 0.1;
        const scrollY = window.scrollY;
        
        // Calculate transform based on scroll position and speed
        const translateY = scrollY * speed;
        
        // Apply with a slight delay for smoother effect
        requestAnimationFrame(() => {
            element.style.transform = `translateY(${translateY}px)`;
        });
    });
} 