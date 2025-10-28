// Seleciona os elementos do HTML que vamos usar
const imageUpload = document.getElementById('imageUpload');
const statusElement = document.getElementById('status');
const refImage = document.getElementById('refImage');

// Caminho para a pasta de modelos
const MODEL_URL = './models';

// Esta variável vai guardar o "descritor facial" da Pessoa X
let referenceDescriptor;

// Define o limite de distância. 
// Valores menores que isso são considerados o "mesmo rosto".
// Você pode ajustar este valor (ex: 0.45, 0.5) para ficar mais ou menos rígido.
const FACE_MATCH_THRESHOLD = 0.48;

// --- FUNÇÃO PRINCIPAL: É EXECUTADA QUANDO A PÁGINA ABRE ---
async function initialize() {
    try {
        // 1. Carrega os modelos de IA
        // - tinyFaceDetector: Detecta onde estão os rostos
        // - faceLandmark68TinyNet: Encontra os pontos (olhos, nariz, boca)
        // - faceRecognitionNet: Calcula o descritor facial (o "vetor" de 128 números)
        updateStatus('Carregando modelos de IA...', 'loading');
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
            faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
            faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
        ]);

        // 2. Calcula o descritor da imagem de referência (Pessoa X)
        updateStatus('Analisando foto de referência...', 'loading');
        referenceDescriptor = await getReferenceDescriptor();

        // 3. Informa que o sistema está pronto
        updateStatus('Pronto! Envie uma foto para verificar.', 'neutral');
        
        // 4. Ativa o botão de upload de imagem
        imageUpload.addEventListener('change', handleImageUpload);

    } catch (error) {
        console.error(error);
        updateStatus('Erro ao inicializar. Recarregue a página.', 'error');
    }
}

// --- FUNÇÃO PARA PEGAR O DESCRITOR DA PESSOA X ---
async function getReferenceDescriptor() {
    try {
        // Detecta um único rosto na imagem de referência
        const detection = await faceapi.detectSingleFace(refImage, new faceapi.TinyFaceDetectorOptions())
                                     .withFaceLandmarks(true) // usa a versão "tiny" dos landmarks
                                     .withFaceDescriptor();
        
        if (!detection) {
            throw new Error('Nenhum rosto encontrado na foto de referência.');
        }
        
        // Retorna apenas o descritor (o vetor de números)
        return detection.descriptor;

    } catch (error) {
        console.error(error);
        updateStatus('Erro ao processar a foto de referência.', 'error');
        return null;
    }
}

// --- FUNÇÃO EXECUTADA QUANDO O USUÁRIO ENVIA UMA IMAGEM ---
async function handleImageUpload(event) {
    if (!event.target.files.length) {
        return;
    }

    // Pega o arquivo enviado
    const file = event.target.files[0];

    // Converte o arquivo em um elemento de imagem HTML
    const image = await faceapi.bufferToImage(file);

    updateStatus('Analisando nova foto...', 'loading');

    try {
        // Detecta o rosto na imagem enviada
        const detection = await faceapi.detectSingleFace(image, new faceapi.TinyFaceDetectorOptions())
                                     .withFaceLandmarks(true)
                                     .withFaceDescriptor();

        if (!detection) {
            updateStatus('Nenhum rosto detectado na foto enviada.', 'error');
            return;
        }

        // 3. Compara os descritores
        // Calcula a "distância euclidiana" entre o rosto da referência e o novo rosto
        const distance = faceapi.euclideanDistance(referenceDescriptor, detection.descriptor);

        // 4. Mostra o resultado
        if (distance < FACE_MATCH_THRESHOLD) {
            updateStatus(`É a Pessoa X! (Distância: ${distance.toFixed(3)})`, 'success');
        } else {
            updateStatus(`NÃO é a Pessoa X. (Distância: ${distance.toFixed(3)})`, 'error');
        }
        
        // Limpa o input para permitir o envio da mesma foto novamente (útil para testes)
        imageUpload.value = '';

    } catch (error) {
        console.error(error);
        updateStatus('Erro ao analisar a imagem.', 'error');
    }
}

// --- FUNÇÃO UTILITÁRIA PARA ATUALIZAR MENSAGENS ---
function updateStatus(message, type) {
    statusElement.innerText = message;
    statusElement.className = type; // Aplica a classe CSS (loading, success, error)
}

// Inicia todo o processo
initialize();