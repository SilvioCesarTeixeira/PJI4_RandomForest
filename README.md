# PJI4_RandomForest
Este repositório contém os arquivos para executar o modelo de ML baseado em Random Forest para previsão de casos de SRAG (Síndrome Respiratória Aguda Grave)
na subprefeitura de Vila Maria/Vila Guilherme, onde se localiza a UBS da Vila Ede.

As variáveis de entrada são:

PM25: material particulado de 2,5 microns, parte por milhão (float)

O3: ozônio presente, parte por milhão (float)

Dias: data convertida em valor numérico, baseado na diferença de dias entre a data específica e o dia 01/01/1900 (inteiro)

A saída do modelo gera uma previsão de quantidade de internações de SRAG (float)

O arquivo 'modelo_random_forest.pkl' é um pacote pickle do modelo já treinado.
Atente para o conteúdo do arquivo 'requirements.txt' com as dependências do modelo, que devem estar instaladas antes da execução.


O script 'Testar_modelo.py' pode ser incorporado ao código do back-end para execução do modelo de predição. É importante que o arquivo 'modelo_random_forest.pkl' esteja no mesmo diretório no backend, bem como atender às dependências do arquivo 'requirements.txt'.


Para que o script 'Testar_modelo.py' seja executado, a linha 9 deve ter o caminho do dataset de entrada (neste caso, um arquivo excel).


Para mudar o arquivo para .csv, basta utilizar o código pd.read_csv('caminho_do_arquivo_.csv').


O importante é que o arquivo contenha uma estrutura semelhante a essa:

PM25    O3    Dias

44.0    	30.0	    44999

51.0  	21.0  	44998

52.252873563218394	16.0	44990

38.0	  25.0	  44984

As saídas do modelo estarão listadas em 'y_pred', que contém todos os valores preditos na mesma ordem de entrada das variáveis PM25, O3 e Dias.

'y_pred' é um array com todos os valores calculados como previsão pelo modelo e tem a seguinte aparência:

array([ 5.80863222,  3.33325228,  4.90881459,  4.46709726,  4.02490881,
        2.87993921,  5.51419453,  5.89306991,  3.0149696 ,  2.35331307,
        2.66106383,  1.51887538,  2.82721884,  3.0343769 ,  1.88218845,
        2.54940729,  2.34331307,  3.7343769 ,  5.00200608,  3.12272036,
        2.17384498,  3.09887538,  3.03775076,  3.16218845,  3.03887538])
        
 Basta salvar este array, arrendondar os valores e utilizá-los para a análise necessária.

