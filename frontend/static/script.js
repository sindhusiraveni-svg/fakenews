function checkNews() {
    const newsText = document.getElementById('newsInput').value;
    const resultDiv = document.getElementById('resultDiv');
    
    if (newsText.trim() === "") {
        resultDiv.innerHTML = "Please enter some text";
        return;
    }

    resultDiv.innerHTML = "Checking...";

    fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ news: newsText })
    })
    .then(res => res.json())
    .then(data => {
        resultDiv.innerHTML = `
            <h3>Result: ${data.prediction}</h3>
            <p>Confidence: ${data.confidence}%</p>
            <p>Model Accuracy: ${data.accuracy}%</p>
            <p>F1-Score: ${data.f1_score}%</p>
        `;
    })
    .catch(err => {
        resultDiv.innerHTML = "Error: " + err;
        console.error('Error:', err);
    });
}


 




