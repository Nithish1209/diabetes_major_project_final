function startPreprocess() {
    const button = document.getElementById('preprocess-button');
    button.classList.add('processing');
    fetch('/preprocess', {
        method: 'POST'
    }).then(response => response.json())
      .then(data => {
          alert(data.message);
          button.classList.remove('processing');
      });
}

function calculateAccuracies() {
    const button = document.getElementById('accuracy-button');
    button.classList.add('processing');
    fetch('/calculate_accuracies', {
        method: 'POST'
    }).then(response => response.json())
      .then(data => {
          let output = '';
          for (const [model, accuracy] of Object.entries(data)) {
              output += `${model}: ${accuracy}%<br>`;
          }
          document.getElementById('prediction-output').innerHTML = output;
          button.classList.remove('processing');
      });
}

function clearForm() {
    document.getElementById('input-form').reset();
    document.getElementById('prediction-output').innerHTML = '';
    document.getElementById('diet-plan').style.display = 'none';
}

document.getElementById('input-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const button = document.getElementById('predict-button');
    button.classList.add('processing');

    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData.entries());

    fetch('/predict', {
        method: 'POST',
        body: new URLSearchParams(data)
    }).then(response => response.json())
      .then(data => {
          document.getElementById('prediction-output').innerHTML = `Prediction: ${data.outcome}`;
          if (data.outcome === "Diabetic") {
              document.getElementById('diet-plan').style.display = 'block';
          } else {
              document.getElementById('diet-plan').style.display = 'none';
          }
          button.classList.remove('processing');
      });
});
