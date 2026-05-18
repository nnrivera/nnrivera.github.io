/* Make it GLOBAL */
window.SPEAKERS = {};

/* Helper */
function addSpeaker(id, data) {
  window.SPEAKERS[id] = data;
}

/* -------- SPEAKERS -------- */

addSpeaker("s1", {
  name: "Andrés Ochoa",
  affiliation: "Escuela de Estadística. Universidad del Valle. Colombia",
  img: "images/andres_ochoa.jpg",
  title: "Un enfoque flexible para la regresión PLS utilizando modelos GAMLSS", 
  abstract: "En este trabajo se propone un nuevo modelo PLS (Partial Least Square) basado en los modelos aditivos generalizados para locación, escala y forma (GAMLSS). Los modelos GAMLSS ofrecen una gran versatilidad para la modelacion estadística, puesto que permiten trabajar con variables respuesta en distintos dominios $(\mathbb{R}, \mathbb{R}^{+}, (0,1))$. Además, permiten trabajan con variables respuesta que tengan distintas distribuciones e incluir el enfoque no paramétrico dentro la modelación utilizando splines, b-splines, entre otros. En este proyecto sea desea combinar dos enfoques de modelación, por un lado los modelos GAMLSS que permiten trabajar con varias distribuciones de probabilidad en la variable respuesta y por otro lado, los modelos PLS, los cuales solucionan problemas de multicolinealidad, datos faltantes y son recomendados en casos de más variables que muestra ($p > n$ ). De esta forma, en este proyecto se propone trabajar el modelo PLS-GAMLSS y comparar su ajuste vs el modelo tradicional PLS.",
  area: "ME"
});

addSpeaker("s2", {
  name: "Joaquin Cavieres",
  affiliation: "Universidad de Stuttgart",
  img: "images/joaqun_cavieres.jpg",
  title: "regTPS-KLE: A novel approach to approximate a Gaussian random field for Bayesian spatial modeling",
  abstract: "Gaussian random field is a ubiquitous model for spatial phenomena in diverse scientific disciplines. Its approximation is often crucial for computational feasibility in simulation, inference, and uncertainty quantification. The Karhunen–Loève Expansion provides a theoretically optimal basis for representing a Gaussian random field as a sum of deterministic orthonormal functions weighted by uncorrelated random variables. While this is a well-established method for dimension reduction and approximation of (spatial) stochastic processes, its practical application depends on the explicit or implicit definition of the covariance structure. In this work, we propose a novel approach, referred to as regTPS-KLE, for approximating a Gaussian random field by explicitly constructing its covariance via a regularized thin plate spline (TPS) kernel. Because TPS kernels are conditionally positive definite and lack a direct spectral decomposition, we formulate the covariance as the inverse of a regularized elliptic operator. To evaluate its statistical performance, we compare its predictive accuracy and computational efficiency with a Gaussian random field approximation constructed using the stochastic partial differential equations (SPDE) method and implemented within an MCMC algorithm. In simulation studies, the predictive differences between the SPDE and regTPS-KLE models were minimal when the spatial field was generated using Matérn and exponential covariance functions, while regTPS-KLE models consistently outperformed the SPDE approach in terms of computational efficiency. In a real data application, regTPS-KLE exhibits superior predictive accuracy compared with SPDE models based on leave-one-out cross-validation while also achieving improved computational efficiency.",
    area: "ME"
});

addSpeaker("s3", {
  name: "Patricio Riquelme",
  affiliation: "Instituto de Estadística, Universidad de Valparaíso",
  img: "",
  title: "Construction of a Random Field with Weibull Distribution using a Clayton-like Space Copula",
  abstract: "We propose a non-Gaussian random field model with a Weibull marginal distribution, constructed using a Clayton-type spatial copula. This approach addresses the need to model spatial data—such as engineering data—that exhibit asymmetry and positive values, properties not adequately captured by traditional Gaussian models. The methodology involves constructing a random field with a Weibull marginal distribution using an emerging class of spatial copulas known as Clayton copulas, to introduce asymmetric dependencies and improve the model’s fit in spatial contexts. Through simulation studies, the efficiency of parameter estimation using weighted composite likelihood is evaluated, highlighting the model’s performance compared to conventional alternatives.",
    area: "ME"
});

addSpeaker("s4", {
  name: "Francisco Plaza Vega",
  affiliation: "Universidad de Santiago de Chile",
  img: "images/francisco_plaza.png",
  title: "Towards quantifying environmentresource interactions using Statistical and Deep Learning tools in a Changing Climate",
  abstract: "The study of fisheries in Chile has consistently shown strong connections with environmental variability, where fluctuations in oceanographic conditions drive shifts in abundance, distribution, and synchronicity across key pelagic species. Identifying ecosystem patterns from long time series of landings and environmental indicators has proven essential to characterize these interactions and to reveal the degree of dependence of fisheries on climatic and oceanographic processes.  Progress has also been made in modeling variability at finer temporal and spatial scales. Statistical frameworks capable of capturing persistence and volatility in fisheries dynamics have been applied to describe anchovy and sardine fluctuations, emphasizing the role of exogenous environmental signals. At the same time, spatio-temporal models have provided insights into how large-scale variability influences the distribution and productivity of other important resources, such as jack mackerel.  In parallel, new opportunities have emerged from artificial intelligence. Computer vision and deep learning tools are being developed to enhance fisheries monitoring, enabling the automatic identification of species with high accuracy. These approaches strengthen the link between data acquisition and ecosystem-based management, supporting real-time decisions and improved understanding of resourceenvironment dynamics.  Building on these foundations, the Fondecyt Iniciación project seeks to advance towards a more systematic quantification of environmentresource interactions. The first stage focuses on developing spatio-temporal models of fisheries coupled with downscaled environmental information. In subsequent stages, statistical downscaling and deep learning methods will be integrated to produce high-resolution climate scenarios, which will then be linked with fisheries models to explore potential climate change impacts.  This research agenda illustrates the value of combining traditional statistical approaches with modern machine and deep learning tools. By progressively advancing towards the quantification of environmentresource interactions, it provides a framework to understand present dynamics while anticipating the challenges that climate change may pose to fisheries and marine ecosystems.",  area: "EA"
});
addSpeaker("s5", {
  name: "Jorge Arevalo",
  affiliation: "Departamento de Meteorología, Universidad de Valparaíso",
  img: "images/jorge_arevalo.jpeg",
  title: "Estadística, Ciencia de Datos e Inteligencia Artificial en Hidrometeorología: Problemas Abiertos y Oportunidades de Investigación",
  abstract: "La hidrometeorología presenta desafíos científicos donde convergen predicción en sistemas dinámicos complejos, integración de datos heterogéneos, cuantificación de incertidumbre e inferencia bajo observaciones incompletas. Este seminario introduce una línea de investigación que aborda estos problemas mediante herramientas de estadística, ciencia de datos e inteligencia artificial, a través de aplicaciones en pronóstico atmosférico e hidrológico, sensores remotos y sistemas observacionales.  Se presentarán resultados y desafíos metodológicos asociados a cinco líneas de trabajo: downscaling de viento mediante machine learning en SiVAR-Austral; pronóstico hidrológico probabilístico en HidroCL e HidroCL-Estacional; estimación de equivalente en agua de nieve (SWE) desde observaciones pasivas de microondas; y desarrollo de capacidades observacionales avanzadas mediante radar meteorológico. Aunque estas aplicaciones surgen en contextos distintos, comparten problemas estadísticos comunes asociados a aprendizaje con restricciones físicas, fusión de datos multifuente, inferencia en sistemas multiescala, modelación de extremos, validación bajo no estacionariedad y predicción probabilística.  El foco del seminario no estará solo en aplicaciones, sino en las preguntas abiertas que emergen desde estos problemas y que pueden formularse como desafíos de investigación en estadística moderna y ciencia de datos. Entre ellas se incluyen modelos híbridos físico-estadísticos, estimación de incertidumbre en sistemas de aprendizaje, transfer learning bajo cambio de dominio, recuperación de variables geofísicas como problemas inversos, y diseño óptimo de observaciones.",
    area: "EA"
});

addSpeaker("s6", {
  name: "Alvaro Figueroa",
  affiliation: "Universidad de Santiago de Chile",
  img: "images/alvaro_figueroa.jpg",
  title: "Modelos de Clasificación Diagnóstica para reducir el Riesgo y la Deserción Académica",
  abstract: "La evaluación diagnóstica en educación superior constituye una herramienta relevante para orientar ajustes curriculares y estrategias de nivelación que favorezcan el desarrollo académico de los estudiantes. En este marco, la presente investigación propone una evaluación diagnóstica para quienes ingresan a Pedagogía en Matemática, basada en Modelos de Clasificación Diagnóstica (MCD), con el propósito de identificar competencias iniciales y caracterizar perfiles de habilidades latentes vinculados con la formación inicial docente.  Se utiliza el Modelo de Clasificación Diagnóstica Mixto, que permite analizar tanto la interacción conjunta de múltiples habilidades como sus efectos aditivos sobre la probabilidad de respuesta correcta en los ítems, reduciendo el número de parámetros estimadoss. Este enfoque contribuye a formular recomendaciones más precisas para la enseñanza y la nivelación, y a sostener la validez de las interpretaciones al vincular explícitamente las respuestas observadas con los atributos evaluados.  El análisis se realizó en RStudio, mediante el paquete CDM y el criterio del menor RMSEA, sobre una muestra seleccionada por conglomerados. Se consideraron siete habilidades latentes: TICs, Aritmética, Álgebra, Función, Geometría, Estadística y Análisis Crítico. Los resultados evidencian fortalezas en Aritmética y Análisis Crítico, y debilidades importantes en Geometría, Estadística, Álgebra y uso académico de TICs. Además, el análisis de regresión sugiere que el número y tipo de habilidades latentes dominadas al inicio del curso pueden constituir un indicador relevante de riesgo académico, especialmente en asignaturas que demandan un alto nivel de desarrollo. Estos hallazgos aportan evidencia útil para el diseño de programas remediales y el fortalecimiento de la formación docente inicial.",
    area: "EA"
});

addSpeaker("s7", {
  name: "Milagros García",
  affiliation: "Instituto de Estadística, Universidad de Valparaíso.",
  title: "Medición de la Alfabetización Estadística: Diseño y Validación de un Instrumento",
  img: "images/milagros_garcia.webp",
  abstract: "Más allá de su concepción pedagógica tradicional, la alfabetización estadística se erige hoy como un constructo multidimensional que exige protocolos de medición de alta precisión psicométrica. El presente estudio examina el panorama científico de este campo mediante una revisión bibliométrica de la producción indexada en Scopus entre 1977 y 2025, con el fin de fundamentar la evaluación de este constructo en la población universitaria panameña. <br> Metodológicamente utilizando herramientas de mapeo bibliométrico (VOSviewer) sobre un corpus de 362 publicaciones, se identificaron tres ejes temáticos dominantes: 1) El desarrollo de competencias en estudiantes y docentes, 2) La conceptualización del razonamiento estadístico, y 3) La arquitectura de los instrumentos de evaluación. Geográficamente, el análisis de red revela una hegemonía en la producción liderada por Estados Unidos, España y Brasil, quienes marcan la pauta en la estandarización de este campo evidenciando la necesidad de generar evidencia local. <br> La investigación se focaliza en un análisis crítico de 45 instrumentos específicos, donde se observó una transición metodológica en la estimación de la fiabilidad: el abandono paulatino del Alfa de Cronbach (α) en favor del Coeficiente Omega (ω) Este cambio se justifica por la capacidad del estadístico ω para gestionar el incumplimiento del principio de tau-equivalencia, ofreciendo estimaciones más realistas de la consistencia interna. Asimismo, se destaca la invarianza de medida según el sexo como una frontera investigativa esencial. Este procedimiento, fundamentado en el Análisis Factorial Confirmatorio (AFC) multigrupo, resulta imperativo para descartar sesgos estructurales y asegurar la equidad en la evaluación de competencias. <br> Finalmente, persiste una brecha entre la propuesta teórica y la validación empírica. El futuro del área en Panamá depende de adoptar estándares métricos robustos que garanticen la neutralidad demográfica de evaluación sean estadísticamente válidos y culturalmente neutros. ",
    area: "EA"
});

addSpeaker("s8", {
  name: "Germán Ibacache",
  affiliation: "Instituto de Estadística, Universidad de Valparaíso.",
  img: "images/german_ibacache.jpg",
  title: "Semiparametric Generalized Modeling",
  abstract: "In this work we discuss some aspect of the theory and application in partially varying-coefficient generalized linear model (PVC-GLM). The discussed model is useful for situations where the variable of interest belongs to the exponential family, and is related to other variables through a semiparamteric regression structure with non-linear interactions. The applicability of our proposal is illustrated through a real data set.",
    area: "ME"
});

addSpeaker("s9", {
  name: "Felipe Osorio",
  affiliation: "Instituto de Estadística, Universidad de Valparaíso.",
  img: "images/felipe_osorio.png",
  title: "On the mean-shift outlier model for LAD regression",
  abstract: "The least absolute deviation (LAD) regression technique is well known for providing an estimation mechanism insensitive to outliers in the response variable. However, relatively few studies have highlighted that these models still can be susceptible to influential observations; therefore the development of appropriate diagnostic measures is required. In this talk, we introduce a procedure for detecting outlying observations in LAD regression based on the mean-shift outlier model. This formulation provides a framework for defining standardized residuals, which enables the construction of a QQ-plot with simulated envelope. A noteworthy by-product of this work is the derivation of the gradient statistic for testing linear hypotheses in LAD regression. We illustrate our findings through the analysis of several datasets commonly examined in the literature.",
    area: "ME"
});

addSpeaker("s10", {
  name: "Tamara Fernández",
  affiliation: "Facultad de Ingeniería y Ciencias, Universidad Adolfo Ibañez",
  img: "images/tamara_fernandez.jpg",
  title: "TBA",
  abstract: "TBA",
    area: "ME"
});

addSpeaker("s11", {
  name: "William Canales",
  affiliation: "Instituto de Estadística, Universidad de Valparaíso",
  img: "images/william_canales.png",
  title: "Métodos de muestreo estadístico para mejorar los planes de examen administrativo y la detección de valores extremos",
  abstract: "Esta investigación propone integrar el muestreo estadístico con la Teoría de Valores Extremos para mejorar la eficiencia de los planes de examen administrativo en la detección de observaciones anómalas y casos críticos. Para ello, desarrolla una nueva familia de estimadores robustos del índice de cola, basados en transformaciones logarítmicas y en el uso de la mediana, que superan limitaciones de estimadores clásicos como Hill, t-Hill y Hill generalizado en términos de sesgo, error cuadrático medio y sensibilidad a outliers, especialmente en muestras pequeñas y contaminadas. A partir de una estimación más estable de la pesadez de cola, se propone además un algoritmo que rediseña dinámicamente los planes de revisión según la estructura extrema de la población. Los resultados preliminares muestran mejoras significativas en precisión y capacidad de detección, aportando tanto al desarrollo teórico de la EVT como a su aplicación práctica en contextos administrativos.",
    area: "ME"
});

addSpeaker("s12", {
  name: "Christian Araya",
  affiliation: "Universidad de Valparaíso y Pontificia Universidad Católica de Valparaíso.",
  img: "images/cristian_araya.jpg",
  title: "Adaptive Approximation for Stochastic Volterra Equations Driven by RiemannLiouville Fractional noise.",
  abstract: "This work develops an adaptive numerical scheme to simulate finite-time explosion in stochastic Volterra equations with fractional kernels. The method combines an exponential approximation of the singular kernel with an Euler-type discretization, yielding a tractable multifactor representation. An adaptive time-stepping strategy is introduced to accurately capture the rapid growth near blow-up.",
    area: "AE"
});


addSpeaker("s13", {
  name: "Soledad Torres",
  affiliation: "Instituto de Ingeniería Matemática, Universidad de Valparaíso",
  img: "images/soledad_torres.webp",
  title: "Modelos estocásticos derivados del movimiento Browniano fraccionario",
  abstract: "El Movimiento Browniano Fraccionario (fBm) es una generalización que modela sistemas con memoria. Su esencia reside en el exponente de Hurst ($H$), que regula la correlación de sus incrementos. Si $H=0.5$, actúa como un movimiento Browniano clásico; es decir, un proceso sin memoria. Para $H > 0.5$, el sistema muestra persistencia, manteniendo tendencias previas en el tiempo. Para $H < 0.5$, presenta anti-persistencia, con una tendencia constante a revertir a la media. En esta charla exploraremos modelos estadísticos derivados del fBm.",
    area: "AE"
});

addSpeaker("s14", {
  name: "Silfrido Gómez",
  affiliation: "Facultad de Ingeniería, Universidad de Valparaíso",
  img: "",
  title: "Spatio-Temporal weighted regression model with fractional-colored noise: Parameter estimation and consistency",
  abstract: "El modelo de Regresión Ponderada Geográfica y Temporalmente (GTWR) es una técnica local bien establecida para analizar la heterogeneidad espacial y la dependencia temporal en datos georreferenciados. Se reconoce por su capacidad para representar entornos del mundo real. En este estudio, ampliamos el modelo GTWR incorporando ruido espacio-temporal coloreado en el espacio y fraccional en el tiempo. Bajo esta formulación, derivamos el estimador de Mínimos Cuadrados Ponderados (WLS) y establecemos formalmente su tasa de convergencia. Para evaluar el rendimiento del estimador WLS, implementamos un estudio de simulación con cinco escenarios definidos. Los resultados de la simulación indican que los residuos del modelo presentan pequeñas variaciones alrededor de cero, lo que sugiere la precisión del estimador. Finalmente, aplicamos el estimador a datos reales sobre la incidencia de enfermedades respiratorias. El análisis de los residuos en esta aplicación empírica nos permite evaluar la capacidad del modelo para capturar la estructura espacio-temporal de los datos.",
    area: "AE"
});

addSpeaker("s15", {
  name: "Alejandra Christen",
  affiliation: "Instituto de Estadística, Universidad de Valparaíso.",
  img: "images/alejandra_christen.png",
  title: "TBA.",
  abstract: "TBA",
    area: "EA"
});
addSpeaker("s16", {
  name: "Patricio Orio",
  affiliation: "Instituto de Neurociencia, Universidad de Valparaíso.",
  img: "images/patricio_orio.jpg",
  title: "Comportamientos emergentes en redes neuronales y su exploración usando teoría de la información.",
  abstract: "Los sistemas complejos exhiben comportamientos colectivos que surgen de interacciones simples entre sus componentes. Comprender cómo emergen estos comportamientos colectivos en la dinámica del cerebro y circuitos neuronales plantea un desafío significativo en neurociencia, ya que se cree que los comportamientos emergentes (o sus trastornos) subyacen a la conciencia, la conducta y los trastornos cerebrales como la enfermedad de Alzheimer, la demencia o la esquizofrenia.  La teoría de la información proporciona un conjunto de herramientas para evaluar las interdependencias estadísticas de orden superior (HOI, High-order interdependencies): estructuras estadísticas que están presentes en un grupo de variables y que no se pueden describir mediante interacciones por pares. La redundancia (cuando la misma información es proporcionada por múltiples variables aleatorias) y la sinergia (cuando la información está contenida conjuntamente en un grupo de variables pero no por separado en subconjuntos de ellas) coexisten en los sistemas complejos y pueden dotarlos de diferentes capacidades de procesamiento de información. Para comprender cómo estos comportamientos emergentes medibles pueden originarse, mantenerse y contribuir al procesamiento de información de un sistema, estudiamos la emergencia de las HOI en redes simuladas con dinámica basada en la dinámica de circuitos neuronales y patrones de conectividad basados en el conectoma del cerebro humano. Nuestros resultados muestran que las HOI emergen con mayor facilidad en redes con topología de pequeño mundo, y además con una mayor riqueza dinámica. Además usando redes neuronales recurrentes (RNNs) entrenadas para resolver una diversidad de tareas cognitivas, encontramos que las HOI de tipo sinergístico emergen de manera preferente en redes entrenadas para resolver tareas de mayor complejidad.",
    area: "EA"
});

addSpeaker("s17", {
  name: "Milan Stehlik",
  affiliation: "Instituto de Estadística, Universidad de Valparaíso.",
  img: "images/stehlik_milan.webp",
  title: "Revolutions in Neural Networks computing and their Data Science Backgrounds: Introducing SPOCU and DEXPSO for finance  and biological complexities",
  abstract: "In this talk, I will introduce our adaptive transfer function SPOCU (Kiselak et al., 2020), which I developed with collaborators to address the insufficiency of standard transfer functions in properly processing real data flows and the necessity of data science and statistical methodologies.  SPOCU is a revolutionary improvement in the speed and innovation of adaptation strategies, filling a gap in existing technology.  Activation functions are crucial in deep learning for extracting complex data patterns, and traditional functions like ReLU, Selu, among others, have limitations in adapting to specialized tasks. Standard transfer functions have limitations in complex setups, thus necessitating the development of robust approaches like large-scale self-normalizing neural networks. To address this, we propose a novel trainable adaptive activation function based on SPOCU construction.  Dynamical networks face challenges with big and irregular data. Optimal activation function selection and hyperparameter management are crucial. The SPOCU transfer function offers flexibility and superior performance in machine learning tasks (see e.g. Bamimore, 2021). Experimental results show improvements in cancer diagnosis and pollutant adsorption dynamics. Developing adaptive algorithms for hyperparameter selection is essential, and our milestone result of DExPSO (Stehlik et al., 2024) is giving a chance to avoid recurrent premature failures of standard neural networks.  In Dinamarca et al. (2025), we showed how Cobetia bacteria adaptation to large temperature ranges can be modeled by an SPOCU-based neural network. Financial applications will be discussed as well. During the talk, we will discuss how the SPOCU prototype adaptive function has been created and explore new ideas for optimizing hyperparameters in adaptive transfer functions like SPOCU for real-world data flows, improving methodologies in different application areas.",
    area: "EA"
});

addSpeaker("s18", {
  name: "Nicolás Rivera",
  affiliation: "Instituto de Estadística, Universidad de Valparaíso.",
  img: "images/nicolas_rivera.png",
  title: "TBA.",
  abstract: "TBA",
    area: "AE"
});

addSpeaker("s19", {
  name: "Héctor Araya",
  affiliation: "Facultad de Ingeniería y Ciencias, Universidad Adolfo Ibañez.",
  img: "images/hector_araya.webp",
  title: "Partially observed Ornstein-Uhlenbeck process driven by small General fractional Gaussian noise.",
  abstract: " The problem of identifying the drift parameter in a partially observed Ornstein--Uhlenbeck process driven by a small general Gaussian noise is considered. This class of Gaussian noises includes several processes, such as fractional Brownian motion, sub-fractional Brownian motion, bi-fractional Brownian motion, and generalized sub-fractional Brownian motion, among others. By applying a least-squares-type method together with finite-difference approximations of the unobserved process, two classes of estimators for the drift parameter of the model are obtained. Their convergence is established under suitable conditions. Some simulations are also presented to illustrate the behavior of these estimators.",
    area: "AE"
});

addSpeaker("s20", {
  name: "Milan Stehlik",
  affiliation: "Instituto de Estadística, Universidad de Valparaíso.",
  img: "",
  title: "TBA",
  abstract: "TBA",
    area: ""
});

addSpeaker("TBA", {
  name: "TBA",
  affiliation: "TBA",
  img: "",
  title: "TBA.",
  abstract: "TBA"
});



/* -------- SPEAKERS GRID -------- */

function renderSpeakersGrid() {
  const container = document.getElementById("speakers-grid");
const placeholder = "images/placeholder.png";

  const cards = Object.values(SPEAKERS)
    .filter((sp, index, self) => 
      sp && 
      sp.name !== "TBA" && 
      index === self.findIndex(s => s && s.name === sp.name) // Keeps only the first match
    ) 
    .map(sp => `
      <div class="speaker-card">
        <img src="${sp.img ||placeholder}">
        <div class="speaker-overlay">
          <div class="speaker-name">${sp.name || 'Unknown'}</div>
          <div>${sp.affiliation || ''}</div>
        </div>
      </div>
    `);

  container.innerHTML = cards.join("");
}

