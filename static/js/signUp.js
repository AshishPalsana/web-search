document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('myForm');
  if (form) {
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      var emailInput = document.getElementById("username");
      var password1Input = document.getElementById("password1");
      var password2Input = document.getElementById("password2");

      var email = emailInput.value;
      var password1 = password1Input.value;
      var password2 = password2Input.value;

      // Clear previous error messages
      document.getElementById("errorTitle").textContent = "";
      document.getElementById("errorTitle").textContent = "";

      // Email validation
      var emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(email)) {
        document.getElementById("errorTitle").textContent = "Invalid email address";
        return;
      }

      // Password validation
      if (password1 !== password2) {
        document.getElementById("errorTitle").textContent = "Password and Confirm Password not match";
        return;
      }

      // If email and passwords are valid, proceed with API call
      // If email is valid, proceed with API call
      var data = {
        "email": email,
        "password": password2
      };

      // Make the API call to login endpoint
      fetch('/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json', // Set the correct Content-Type header for JSON
        },
        body: JSON.stringify(data),
      })
        .then(function (response) {
          if (response.ok) {
            window.location.href = '/signInPage';
          } else {
            response.json().then(function (data) {
              var errorMessage = data.error || "Sign up failed. Please try again.";
              c
              document.getElementById("errorTitle").textContent = errorMessage
            });
          }
        })
      //  .catch(function(error) {
      //    // Error occurred during API call
      //    // Add your code here to handle the error
      //
      //  });
    });
  }
});
console.log('signup')