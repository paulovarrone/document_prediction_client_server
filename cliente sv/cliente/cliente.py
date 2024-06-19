from flask import Flask, request, redirect, render_template
import requests

app = Flask(__name__)



@app.route('/index')
def index():
    return render_template('index.html')
    

@app.route('/enviar_pdf', methods=['POST'])
def enviar():
    if 'uploaded_file' in request.files:
        file = request.files['uploaded_file']
        files = {'uploaded_file': (file.filename, file.read(), file.content_type)}
        proxy = {
            'http': 'http://SEU PROXY : PORTA',
            'https': 'http://SEU PROXY : PORTA'
        }
        response = requests.post('http://IP DO SERVIDOR:5000/classificar', files=files, proxies=proxy)
        if response.ok:
            result = response.json().get('classification_report', 'No classification result')
            return render_template('index.html', classification_result=result)
        else:
            error_message = 'Failed to classify PDF'
            return render_template('index.html', error_classificacao_message=error_message)
    return render_template('index.html', error_classificacao_message='Falha no envio do arquivo')


@app.route('/corrigir_pdf', methods=['POST'])
def corrigir():
    if 'correcao_file' in request.files:
        file = request.files['correcao_file']
        especializada = request.form['especializada'].upper()
        files = {'correcao_file' : (file.filename, file.read(), file.content_type)}
        data = {'especializada': especializada}
        proxy = {
            'http': 'http://SEU PROXY : PORTA',
            'https': 'http://SEU PROXY : PORTA'
        }

        try:
            response = requests.post('http://IP DO SERVIDOR:5000/ajustar', files=files, data=data, proxies=proxy)
            if response.ok:
                result = response.json().get('classification_report', 'No classification result')
                return render_template('index.html', correcao_ok=result)
            else:
                return render_template('index.html', error_correcao_message='Falha no envio do arquivo')
        except requests.exceptions.RequestException as e:
            return render_template('index.html', error_correcao_message=str(e))
    else:
        return render_template('index.html', error_correcao_message='Nenhum arquivo enviado')



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)