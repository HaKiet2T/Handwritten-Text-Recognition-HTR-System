const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let currentMode = 'single'; // 'single' or 'multi'
let undoStack = [];
let isEraserMode = false;

// Brush settings
let brushSize = 3;
let brushColor = '#000000';

// Initialize canvas
function initCanvas() {
    adjustCanvasSize();
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = brushColor;
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Save initial state
    saveCanvasState();
}

// Adjust canvas size based on screen
function adjustCanvasSize() {
    // Lưu trạng thái canvas hiện tại
    const oldWidth = canvas.width;
    const oldHeight = canvas.height;
    let imageData = null;
    
    if (oldWidth > 0 && oldHeight > 0) {
        try {
            imageData = ctx.getImageData(0, 0, oldWidth, oldHeight);
        } catch(e) {
            console.log('Could not save canvas state');
        }
    }
    
    // Tính toán kích thước mới dựa trên màn hình
    const containerWidth = Math.min(window.innerWidth - 40, 900);
    
    if (window.innerWidth <= 480) {
        // Điện thoại nhỏ
        canvas.width = Math.min(containerWidth, 350);
        canvas.height = Math.round(canvas.width * 0.6);
    } else if (window.innerWidth <= 768) {
        // Tablet/điện thoại lớn
        canvas.width = Math.min(containerWidth, 500);
        canvas.height = Math.round(canvas.width * 0.6);
    } else {
        // Desktop
        canvas.width = 900;
        canvas.height = 600;
    }
    
    updateCanvasContext();
    
    // Vẽ lại nền trắng
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    console.log(`Canvas resized to: ${canvas.width}x${canvas.height}`);
}

// Update canvas context after resize
function updateCanvasContext() {
    ctx.fillStyle = 'white';
    ctx.strokeStyle = isEraserMode ? 'white' : brushColor;
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
}

// Get coordinates from mouse or touch event
function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    if (e.touches) {
        return {
            x: (e.touches[0].clientX - rect.left) * scaleX,
            y: (e.touches[0].clientY - rect.top) * scaleY
        };
    } else {
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }
}

// Start drawing
function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    const coords = getCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(coords.x, coords.y);
}

// Draw
function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    
    const coords = getCoordinates(e);
    ctx.strokeStyle = isEraserMode ? 'white' : brushColor;
    ctx.lineWidth = brushSize;
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
}

// Stop drawing
function stopDrawing(e) {
    if (isDrawing) {
        e.preventDefault();
        isDrawing = false;
        ctx.beginPath();
        saveCanvasState();
    }
}

// Save canvas state for undo
function saveCanvasState() {
    undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    if (undoStack.length > 20) {
        undoStack.shift();
    }
}

// Undo last stroke
function undoStroke() {
    if (undoStack.length > 1) {
        undoStack.pop(); // Remove current state
        const previousState = undoStack[undoStack.length - 1];
        ctx.putImageData(previousState, 0, 0);
    }
}

// Toggle eraser mode
function toggleEraser() {
    isEraserMode = !isEraserMode;
    const eraserBtn = document.getElementById('eraser-btn');
    
    if (isEraserMode) {
        eraserBtn.classList.add('active');
        canvas.classList.add('eraser-mode');
        eraserBtn.textContent = '✏️ Vẽ';
    } else {
        eraserBtn.classList.remove('active');
        canvas.classList.remove('eraser-mode');
        eraserBtn.textContent = '🧹 Tẩy';
    }
}

// Mouse events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch events
canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing, { passive: false });

// Prevent scrolling when touching canvas
document.body.addEventListener('touchstart', function(e) {
    if (e.target === canvas) {
        e.preventDefault();
    }
}, { passive: false });

document.body.addEventListener('touchmove', function(e) {
    if (e.target === canvas) {
        e.preventDefault();
    }
}, { passive: false });

// Brush settings listeners
document.getElementById('brush-size').addEventListener('input', function(e) {
    brushSize = parseInt(e.target.value);
    document.getElementById('brush-size-value').textContent = brushSize + 'px';
    ctx.lineWidth = brushSize;
});

document.getElementById('brush-color').addEventListener('input', function(e) {
    brushColor = e.target.value;
    if (!isEraserMode) {
        ctx.strokeStyle = brushColor;
    }
});

// Clear canvas
function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('result').innerHTML = '<p class="status-message">Vẽ chữ và nhấn "Nhận diện" để xem kết quả</p>';
    document.getElementById('processing-steps').innerHTML = '';
    undoStack = [];
    saveCanvasState();
}

// Update mode
function updateMode() {
    const selected = document.querySelector('input[name="mode"]:checked').value;
    currentMode = selected;
    console.log('Mode changed to:', currentMode);
}

// Toggle toolbar
function toggleToolbar() {
    const toolbar = document.getElementById('toolbar');
    const icon = document.getElementById('toolbar-icon');
    toolbar.classList.toggle('hidden');
    icon.textContent = toolbar.classList.contains('hidden') ? '☰' : '✕';
}

// Predict function
async function predict() {
    const resultDiv = document.getElementById('result');
    const stepsDiv = document.getElementById('processing-steps');
    resultDiv.innerHTML = '<p class="status-message">🔄 Đang xử lý...</p>';
    stepsDiv.innerHTML = '';

    // Tạo canvas tạm để đảm bảo ảnh có kích thước chuẩn
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    
    // Sử dụng kích thước thực của canvas (không phải CSS size)
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    
    // Vẽ nền trắng trước
    tempCtx.fillStyle = 'white';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    
    // Copy nội dung từ canvas gốc
    tempCtx.drawImage(canvas, 0, 0);
    
    const imageData = tempCanvas.toDataURL('image/png');
    const decodeMode = document.getElementById('decode-mode').value;
    const beamWidth = parseInt(document.getElementById('beam-width').value);
    const spellcheck = document.getElementById('spellcheck').checked;
    
    console.log(`Sending image: ${canvas.width}x${canvas.height}, mode: ${currentMode}`);

    try {
        const response = await fetch('/predict_handwriting', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData,
                mode: currentMode,
                decode_mode: decodeMode,
                beam_width: beamWidth,
                spellcheck: spellcheck
            })
        });

        const data = await response.json();
        
        if (data.error) {
            resultDiv.innerHTML = `<p style="color: red;">❌ Lỗi: ${data.error}</p>`;
            return;
        }

        // Display results based on mode
        if (currentMode === 'multi' && data.words) {
            displayMultiWordResult(data);
        } else {
            displaySingleWordResult(data);
        }

        // Display processing steps
        if (data.processing_steps) {
            displayProcessingSteps(data.processing_steps);
        }

    } catch (error) {
        resultDiv.innerHTML = `<p style="color: red;">❌ Lỗi kết nối: ${error.message}</p>`;
        console.error('Prediction error:', error);
    }
}

// Display single word result
function displaySingleWordResult(data) {
    const resultDiv = document.getElementById('result');
    let html = '';

    html += `<div class="recognized-text">"${data.text}"</div>`;
    
    if (data.confidence !== undefined) {
        html += `<p class="confidence-score">Độ tin cậy: <strong>${(data.confidence * 100).toFixed(1)}%</strong></p>`;
    }

    if (data.raw_text && data.raw_text !== data.text) {
        html += `<p style="color: #666; margin-top: 15px;">📝 Gốc: "${data.raw_text}"</p>`;
    }

    resultDiv.innerHTML = html;
}

// Display multi-word result
function displayMultiWordResult(data) {
    const resultDiv = document.getElementById('result');
    let html = '';

    // Show segmentation visualization if available
    if (data.segmentation_image) {
        // Check if image already has data URL prefix
        const imgSrc = data.segmentation_image.startsWith('data:') 
            ? data.segmentation_image 
            : `data:image/png;base64,${data.segmentation_image}`;
        html += `<img src="${imgSrc}" 
                      alt="Segmentation" 
                      class="segmentation-image">`;
    }

    html += `<h3 style="color: #667eea; margin: 20px 0;">📝 Kết quả nhận diện từng từ:</h3>`;

    // Display each word
    data.words.forEach((word, index) => {
        html += `
            <div class="word-result">
                <h3>Từ ${index + 1}</h3>
                <p><strong>Văn bản:</strong> "${word.text}"</p>
                ${word.confidence !== undefined ? 
                    `<p><strong>Độ tin cậy:</strong> ${(word.confidence * 100).toFixed(1)}%</p>` : ''}
                ${word.raw_text && word.raw_text !== word.text ? 
                    `<p><strong>Gốc:</strong> "${word.raw_text}"</p>` : ''}
            </div>
        `;
    });

    // Full text
    const fullText = data.words.map(w => w.text).join(' ');
    html += `<div class="recognized-text" style="margin-top: 20px;">
                📄 Toàn bộ văn bản: "${fullText}"
             </div>`;

    resultDiv.innerHTML = html;
}

// Display processing steps
function displayProcessingSteps(steps) {
    const stepsDiv = document.getElementById('processing-steps');
    let html = '';

    steps.forEach(step => {
        // Check if image already has data URL prefix
        const imgSrc = step.image.startsWith('data:') ? step.image : `data:image/png;base64,${step.image}`;
        
        html += `
            <div class="step-card">
                <img src="${imgSrc}" 
                     alt="${step.name}" 
                     class="step-image">
                <div class="step-title">${step.name}</div>
                ${step.shape ? 
                    `<div class="step-meta">Kích thước: ${step.shape[1]}×${step.shape[0]}</div>` : ''}
            </div>
        `;
    });

    stepsDiv.innerHTML = html;
}

// Initialize on load
window.addEventListener('load', function() {
    initCanvas();
    
    // Ẩn toolbar mặc định trên mobile
    if (window.innerWidth <= 768) {
        const toolbar = document.getElementById('toolbar');
        const icon = document.getElementById('toolbar-icon');
        toolbar.classList.add('hidden');
        icon.textContent = '☰';
    }
});
window.addEventListener('resize', adjustCanvasSize);
