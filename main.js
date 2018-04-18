function logInPage(data){
  let date = new Date().toLocaleString();
  var pageConsole = document.getElementById("model-progress");
  var newData = document.createElement('span');
  newData.innerHTML = "[" + date + "] " + data;
  newData.className = "log-text"
  pageConsole.appendChild(newData);
  pageConsole.scrollTop = pageConsole.scrollHeight;
}

document.addEventListener('DOMContentLoaded', () => {
})