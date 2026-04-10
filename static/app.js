document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const uploadContent = uploadArea.querySelector('.upload-content');
    const loadingState = document.getElementById('loadingState');
    const resultsContainer = document.getElementById('resultsContainer');
    const resetBtn = document.getElementById('resetBtn');
    const resultTemplate = document.getElementById('resultTemplate');

    // Click to upload
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('image/')) {
                handleFile(file);
            } else {
                alert('Vui lòng tải lên một tệp hình ảnh hợp lệ.');
            }
        }
    });

    resetBtn.addEventListener('click', () => {
        resultsContainer.innerHTML = '';
        resultsContainer.classList.add('hidden');
        resetBtn.classList.add('hidden');
        uploadArea.classList.remove('hidden');
        fileInput.value = ''; // Clear input
    });

    async function handleFile(file) {
        // Show loading state
        uploadContent.classList.add('hidden');
        loadingState.classList.remove('hidden');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'Không thể xử lý hình ảnh');
            }

            renderResults(data.results);

        } catch (error) {
            console.error(error);
            alert('Lỗi: ' + error.message);
            // Reset state on error
            uploadContent.classList.remove('hidden');
            loadingState.classList.add('hidden');
        }
    }

    function renderResults(results) {
        // Hide upload area
        uploadArea.classList.add('hidden');
        
        // Reset loading state for next time
        uploadContent.classList.remove('hidden');
        loadingState.classList.add('hidden');

        // Clear previous
        resultsContainer.innerHTML = '';

        if (!results || results.length === 0) {
            const emptyDiv = document.createElement('div');
            emptyDiv.className = 'empty-result';
            emptyDiv.textContent = 'Không tìm thấy biển số Việt Nam nào trong ảnh.';
            resultsContainer.appendChild(emptyDiv);
        } else {
            results.forEach(res => {
                const clone = resultTemplate.content.cloneNode(true);
                clone.querySelector('[data-target="plate"]').textContent = res.plate;
                clone.querySelector('[data-target="city"]').textContent = res.city;
                clone.querySelector('[data-target="old_city"]').textContent = res.old_city;
                resultsContainer.appendChild(clone);
            });
        }

        // Show results and reset button
        resultsContainer.classList.remove('hidden');
        resetBtn.classList.remove('hidden');
    }
});
