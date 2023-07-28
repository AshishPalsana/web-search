document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('myForm');
  if (form) {
    form.addEventListener('submit', function (event) {
      event.preventDefault();

      var emailInput = document.getElementById("email");
      var passwordInput = document.getElementById("password");

      var email = emailInput.value;
      var password = passwordInput.value;

      var emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(email)) {
        document.getElementById("errorTitle").textContent = "Invalid email address";
        return;
      }

      // If email is valid, proceed with API call
      var data = {
        "email": email,
        "password": password
      };

      // Make the API call to login endpoint
      fetch('/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json', // Set the correct Content-Type header for JSON
        },
        body: JSON.stringify(data),
      })
        .then(function (response) {
          console.log("###response", response)
          if (response.ok) {
            window.location.href = '/';
          } else {
            response.json().then(function (data) {
              var errorMessage = data.error || "Login failed. Please try again.";
              document.getElementById("errorTitle").textContent = errorMessage;
            });
          }
        })
    });
  }
});
