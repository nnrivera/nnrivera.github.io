/* Make it GLOBAL */
window.SPEAKERS = {};

/* Helper */
function addSpeaker(id, data) {
  window.SPEAKERS[id] = data;
}

/* -------- SPEAKERS -------- */

addSpeaker("s1", {
  name: "Speaker 1",
  affiliation: "IDEUV",
  img: "images/s1.jpg",
  abstract: "This talk introduces modern statistical learning methods and their theoretical guarantees."
});

addSpeaker("s2", {
  name: "Speaker 2",
  affiliation: "IDEUV",
  img: "images/s2.jpg",
  abstract: "We discuss advances in kernel methods and applications to hypothesis testing."
});

addSpeaker("s3", {
  name: "Speaker3",
  affiliation: "PUCV",
  img: "images/s3.png",
  abstract: "An overview of high-dimensional inference and sparsity-based techniques."
});

addSpeaker("s4", {
  name: "David",
  affiliation: "PUC",
  img: "images/s4.png",
  abstract: "Bayesian approaches to complex models with computational challenges."
});

addSpeaker("s5", {
  name: "Eva",
  affiliation: "Universidad de Chile",
  img: "images/s5.jpg",
  abstract: "Applications of statistical modeling in social sciences and policy."
});

addSpeaker("s6", {
  name: "Franco",
  affiliation: "Universidad de Playa Ancha",
  img: "images/s6.jpg",
  abstract: "Optimization methods in machine learning and large-scale systems."
});

addSpeaker("s7", {
  name: "Pedro",
  affiliation: "IDEUV",
  img: "images/s7.jpg",
  abstract: "Deep learning theory and representation learning."
});

addSpeaker("s8", {
  name: "Speaker 8",
  affiliation: "IDEUV",
  img: "images/s8.jpg",
  abstract: "Stochastic processes and their applications in modern statistics."
});

/* -------- SPEAKERS GRID -------- */

function renderSpeakersGrid() {
  const container = document.getElementById("speakers-grid");

  const cards = Object.values(SPEAKERS)
    .filter(sp => sp)   
    .map(sp => `
      <div class="speaker-card">
        <img src="${sp.img || 'https://via.placeholder.com/300'}">
        <div class="speaker-overlay">
          <div class="speaker-name">${sp.name || 'Unknown'}</div>
          <div>${sp.affiliation || ''}</div>
        </div>
      </div>
    `);

  container.innerHTML = cards.join("");
}

