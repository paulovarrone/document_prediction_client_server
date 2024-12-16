from flask import Flask, request, redirect, render_template
import requests

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')
    

@app.route('/enviar_pdf', methods=['POST'])
def enviar():
    try:
        if 'uploaded_file' in request.files:
            file = request.files['uploaded_file']
            files = {'uploaded_file': (file.filename, file.read(), file.content_type)}
            proxy = {
                'http': None,
                'https': None
            }
            response = requests.post('http://SEU IP/classificar', files=files, proxies=proxy)
            if response.ok:
                result = response.json().get('classification_report', 'No classification result')
                return render_template('index.html', classification_result=result)
            else:
                error_message = 'Falha ao classificar, utilize um PDF válido'
                return render_template('index.html', error_classificacao_message=error_message)
    except Exception:
        return render_template('index.html', error_classificacao_message='Falha no envio do arquivo')



@app.route('/corrigir_pdf', methods=['POST'])
def corrigir():
    if 'correcao_file' in request.files:
        file = request.files['correcao_file']
        especializada = request.form['especializada'].upper()
        files = {'correcao_file' : (file.filename, file.read(), file.content_type)}
        data = {'especializada': especializada}
        proxy = {
            'http': None,
            'https': None
        }

        if especializada not in ['PAS', 'PDA', 'PPE', 'PSE', 'PTR', 'PUMA', 'PTA']:
            return render_template('index.html', error_correcao_message='ESPECIALIZADA DIGITADA ERRADA')

        try:
            response = requests.post('http://SEU IP/ajustar', files=files, data=data, proxies=proxy)
            if response.ok:
                result = response.json().get('classification_report', 'No classification result')
                return render_template('index.html', correcao_ok=result)
            else:
                return render_template('index.html', error_correcao_message='Falha no envio do arquivo')
        except requests.exceptions.RequestException as e:
            return render_template('index.html', error_correcao_message=str(e))
    else:
        return render_template('index.html', error_correcao_message='Nenhum arquivo enviado')
    

@app.route('/treinamento', methods=['POST'])
def treinar():
    try:

        proxy = {
            'http': None,
            'https': None
        }

        response = requests.post('http://SEU IP/treino', proxies=proxy)

        if response.ok:
            return render_template('index.html', treino_ok = 'IA treinada com sucesso')
        else:
            return render_template('index.html', erro_treino='A IA não pode ser treinada')

    except Exception:
        return render_template('index.html', erro_treino='ERRO ao treinar IA')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)


