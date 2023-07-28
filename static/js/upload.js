document.addEventListener('DOMContentLoaded', function () {
  const formOnSearchPage = document.getElementById('audio-form');
  const formOnResultPage = document.getElementById('audio-form-result');
  const formInput = document.getElementById("file-upload-input");
  const resultsEl = document.getElementById("results");

  if (formInput) {
    formInput.addEventListener("change", (e) => {
      const fileName = e.target.files[0].name;
      document.getElementById("file-name").textContent = `File name: ${fileName}`
      localStorage.setItem("fileName",fileName)
    })
  }
  if (formOnSearchPage) {
    if (!resultsEl) {
      localStorage.removeItem("data");
        localStorage.removeItem("fileName")
    }
    formOnSearchPage.addEventListener('submit', function (event) {
      event.preventDefault();

      const formUploadInput = document.getElementById('file-upload-input');
      const file = formUploadInput.files[0]
      const fileName = formUploadInput.files[0].name

      var data = new FormData()
      data.append('file', file)
      data.append("name", fileName)

      // Make the API call to login endpoint
      const loading = document.getElementById("loading");
      loading.style.display = "block"

      fetch('/upload', {
        method: 'POST',
        body: data,
      })
        .then((response) => response.json())
        .then(res => {
          localStorage.setItem("data", JSON.stringify(res.data))
          window.location.replace("/resultsPage")

        }).catch(e => {
          console.log("e", e)
        })
        .finally(() => {
          loading.style.display = "none"
        })
    });
  }
  if (resultsEl) {
    const localData = localStorage.getItem("data");
    const fileName = localStorage.getItem("fileName")
    if(fileName){
        document.getElementById("file-name-span").textContent = `: ${fileName}`
    }
    if (localData) {
      const data = JSON.parse(localStorage.getItem("data"))

      console.log("data-->", { data, type: typeof data })

      let el = ""
      data.forEach(d => {
        el += `
        <div class="result">
            <h2>${d.name}</h2>
            <p class="duration">Duration: 3:00</p>
            <img src="static/img/Triangle@3x.png" alt="Play Button" class="preview" onclick="playAudio('${d.name}')" id="img-${d.name}" />
            <a href="https://mp3.chillhop.com/serve.php/?mp3=27485" download>
              <img src="static/img/download@3x.png" alt="Download Button" class="download" />
            </a>
            <audio id="${d.name}" src="https://mp3.chillhop.com/serve.php/?mp3=27485"></audio>
        </div>
        `
      })
      resultsEl.innerHTML = el;
    } else {
      resultsEl.innerHTML = `<p align="center">No record found</p>`
    }

  }
});

