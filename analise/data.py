# -*- coding: utf-8 -*-

# ==============================================================================
# Comentários, etc, etc...
# ==============================================================================

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

# ==============================================================================

DATASET_DIR = "../dataset/"

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
SOIL_FILENAME = "soil_data.csv"

FIELD_FILENAMES = [("field-%d.csv" % i) for i in range(28)]

# ==============================================================================

def load_dataset():
    # Carrega os dados
    train_data = pd.read_csv(DATASET_DIR + TRAIN_FILENAME)
    test_data = pd.read_csv(DATASET_DIR + TEST_FILENAME)
    soil_data = pd.read_csv(DATASET_DIR + SOIL_FILENAME)

    field_data = []
    for filename in FIELD_FILENAMES:
        field_data.append(pd.read_csv(DATASET_DIR + filename))

    return train_data, test_data, soil_data, field_data

# ==============================================================================

def generate_dates(dataframe, year_str, month_str):
    dataframe["date"] = pd.to_datetime(
        (dataframe[year_str] * 10000 + dataframe[month_str] * 100 + 1).apply(str),
        format="%Y%m%d")

# ==============================================================================

def generate_acc_precipitation(dataframe):
    dataframe["acc_precipitation"] = dataframe["Precipitation"].rolling(4, min_periods=1).mean()

# ==============================================================================

def generate_monolith_train(train_data, soil_data, field_data):
    # Coloca o número do field em uma coluna em cada data frame
    for i in range(len(field_data)):
        field_data[i]["field"] = i
    full_field_data = pd.concat(field_data)

    # Cola todos os datasets
    join_field = train_data.merge(full_field_data, how="inner", left_on=["field", "date"], right_on=["field", "date"])
    full_data = join_field#.merge(soil_data, how="inner", left_on=["field"], right_on=["field"])

    # Remove atributos redundantes
    full_data.drop("harvest_year", axis=1, inplace=True)
    full_data.drop("harvest_month", axis=1, inplace=True)

    # Reordena de acordo com o id
    full_data.set_index("Id", inplace=True)
    full_data.sort_values(by=["Id"], inplace=True)

    full_data.to_csv("monolith.csv", index=True)

    return full_data

# ==============================================================================

def main():
    # Carrega os dados
    train_data = pd.read_csv(DATASET_DIR + TRAIN_FILENAME)
    test_data = pd.read_csv(DATASET_DIR + TEST_FILENAME)
    soil_data = pd.read_csv(DATASET_DIR + SOIL_FILENAME)

    field_data = []
    for filename in FIELD_FILENAMES:
        field_data.append(pd.read_csv(DATASET_DIR + filename))

    # Corrige o campo de mês e ano no train_data e no test_data
    generate_dates(train_data, "harvest_year", "harvest_month")
    generate_dates(test_data, "harvest_year", "harvest_month")

    # Corrige o campo de mês e ano nos field_data[...]
    for n_field_data in field_data:
        generate_dates(n_field_data, "year", "month")
        generate_acc_precipitation(n_field_data)

    print("--- ANÁLISE DOS DADOS DE TREINAMENTO E DE TESTE ---\n")

    # RELAÇÃO DO ATRIBUTO PRODUCTION COM OS OUTROS

    # Atributo 'production' vs 'date'
    sns.lineplot(x='date', y="production", data=train_data)
    plt.show()
    # Nota-se que a produção tem picos por volta de outubro e vales por volta de março.
    # Além disso, nota-se uma tendência geral de crescimento na produção: os picos estão mais altos a cada ano.

    # Atributo 'production' vs 'age'
    sns.lineplot(x='age', y="production", data=train_data)
    plt.show()
    # Nota-se que há um pico de produção para árvores por volta dos 15 anos de idade.
    # Árvores com 16 ou mais anos possuem produção com tendência constante.
    # Árvores jovens (menos de 5 anos) possuem baixa produtividade, e essa produtividade cresce ao passar dos anos

    # Atributo 'production' vs 'harvest_month'
    sns.lineplot(x='harvest_month', y="production", data=train_data)
    plt.show()
    # Neste gráfico, é bastante evidente o padrão que se "repete" ao passar dos anos.
    # Há um pico de produtividade bastante intenso no mês de outubro.
    # A produção é relativamente baixa (em média) de fevereiro a junho.

    # Atributo 'production' vs 'type'
    sns.barplot(x='type', y='production', data=train_data)
    plt.show()
    # Esse gráfico apresenta a média de produtividade para os diferentes tipos de árvore.
    # Nota-se que as árvores de tipo 4 e 2 são relativamente mais produtivas que as outras.
    # As árvores de tipo 6 e 5 (as mais comuns) têm produtividade média.
    # As árvores de tipo 1 e 3 produzem um pouco menos que as 6 e 5.
    # As árvores de tipo 0 produzem muito pouco em comparação com as outras.

    # CDF da produção
    sns.distplot(train_data["production"], hist_kws={'cumulative': True}, kde_kws={'cumulative': True})
    plt.show()
    # A partir desse gráfico, pode-se notar que aproximadamente 90% dos dados de produção estão abaixo de 0.4,
    # e aproximadamente 80% estão abaixo de 0.25.

    # Atributo 'type'
    print("Atributo 'type' do train.csv (contagem): ")
    print(train_data["type"].value_counts())
    print()
    print("Atributo 'type' do test.csv (contagem): ")
    print(test_data["type"].value_counts())
    print()
    # O tipo mais frequente é 5, tanto no conjunto de treinamento quanto no de teste,
    # representando 81.1% dos dados de treinamento e 77.6% dos dados de teste.
    # Os valores 2, 3 e 4 são os próximos mais frequentes, também em frequências
    # semelhantes em ambos os arquivos. O 'type' 0, no entanto, aparece com frequência
    # significativamente maior nos dados de teste. Além disso, há dois valores que não
    # aparecem nos dados de treinamento, mas aparecem nos dados de teste: -1 e 7.

    print("Listagem dos 'type' para cada field:")
    for i in range(len(field_data)):
        print("Field " + str(i))
        print(train_data[train_data["field"] == i]["type"].value_counts())
        print(test_data[test_data["field"] == i]["type"].value_counts())
        print()
    """
    Lista dos 'types' que aparecem em cada field:
    Field  0: 5, 2, 3
    Field  1: 5, 2, 3
    Field  2: 5, 2, 3
    Field  3: 5, 6
    Field  4: 5
    Field  5: 5
    Field  6: 5
    Field  7: 5
    Field  8: 5, 1, 0, -1
    Field  9: 5, 0, -1, 7
    Field 10: 5
    Field 11: 5
    Field 12: 5
    Field 13: 5
    Field 14: 5, 4
    Field 15: 5
    Field 16: 5, 4
    Field 17: 5
    Field 18: 5
    Field 19: 5
    Field 20: 5
    Field 21: 5
    Field 22: 5
    Field 23: 5
    Field 24: 5
    Field 25: 5, -1
    Field 26: 5, 4, -1
    Field 27: 5
    
    Nota-se que o 'type' 5 aparece em todos os fields.
    Os 'type' 2 e 3 sempre aparecem juntos, e apenas nos fields 0, 1 e 2.
    O 'type' 4 apenas aparece nos fields 14, 16 e 26.
    O 'type' -1 aparece nos fields 8, 9, 25 e 26.
    O 'type' 0 aparece nos fields 8 e 9.
    O 'type' 6 apenas aparece no field 3.
    O 'type' 1 apenas aparece no field 8.
    O 'type' 7 apenas aparece no field 9.
    Os fields 4, 5, 6, 7, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24 e 27
    possuem apenas entradas com 'type' 5.
    """

    # Faz join de todos os dados (retirar o comentário para gerar novamente)
    generate_monolith_train(train_data, soil_data, field_data)

    # Atributo 'Production'
    print("Atributo 'production' do train.csv (média): ")
    print(train_data["production"].mean())

    print("Média da produção para cada field:")
    field_index = []
    prod_mean_by_field = []
    for i in range(len(field_data)):
        field_index.append(i)
        prod_mean_by_field.append(train_data[train_data["field"] == i]["production"].mean())
        print("Field " + str(i) + ": " + str(train_data[train_data["field"] == i]["production"].mean()))
    sns.barplot(x=field_index, y=prod_mean_by_field)
    plt.xlabel("field")
    plt.ylabel("avg_production")
    plt.show()
    # A partir desse gráfico, nota-se que os fields 12 e 14 (e 16) se destacam por ter uma média de produção maior
    # que os outros. Os fields 8 e 9 (e 5), por outro lado, destacam-se por terem média de produção muito inferior
    # aos demais.

    sns.boxplot(x="field", y="production", data=train_data)
    plt.show()
    # Esta é uma forma de visualizar a distribuição probabilística das produções por field. Neste gráfico, pode-se ver
    # que as ocorrências de produção alta são, em geral, consideradas outliers. A maior parte dos pontos se localiza na
    # região de 0.1 a 0.25 de produtividade.

    sns.boxplot(x="field", y="production", data=train_data[train_data["harvest_month"].isin([1, 2, 3, 4, 5, 6])])
    plt.show()
    # A distribuição é totalmente deslocada para baixo no primeiro semestre do ano

    sns.boxplot(x="field", y="production", data=train_data[train_data["harvest_month"].isin([7, 8, 9, 10, 11, 12])])
    plt.show()
    # No segundo semestre, é deslocada para cima

    sns.boxplot(x="age", y="production", data=train_data)
    plt.show()
    # Aqui é mostrada a distribuição da produção das plantas de acordo com a idade

    sns.boxplot(x="harvest_month", y="production", data=train_data)
    plt.show()

    # O arquivo soil_data.csv introduz uma grande quantidade de atributos.
    # É necessário identificar se realmente são úteis.

    # --- Cálculo da correlação entre atributos do solo ---

    correlation = soil_data[
        ["BLDFIE_sl1", "BLDFIE_sl2", "BLDFIE_sl3", "BLDFIE_sl4", "BLDFIE_sl5", "BLDFIE_sl6", "BLDFIE_sl7"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()
    # Os atributos BLDFIE têm alta correlação entre o sl1 e sl2, e entre os sl4 a sl7. O sl3 não tem muita correlação
    # com os outros. Pode-se reduzir esses dados apenas para as colunas BLDFIE_sl1, BLDFIE_sl3 e BLDFIE_sl7.

    correlation = soil_data[
        ["CECSOL_sl1", "CECSOL_sl2", "CECSOL_sl3", "CECSOL_sl4", "CECSOL_sl5", "CECSOL_sl6", "CECSOL_sl7"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()
    # Todos os atributos CECSOL têm alta correlação entre si. O CECSOL_sl4 é uma boa escolha (ponto central)

    correlation = soil_data[
        ["CLYPPT_sl1", "CLYPPT_sl2", "CLYPPT_sl3", "CLYPPT_sl4", "CLYPPT_sl5", "CLYPPT_sl6", "CLYPPT_sl7"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()
    # Os atributos CLYPPT parecem que têm uma divisão mais bipartida. Os sl1 até sl4 têm alta correlação,
    # e os sl4 até sl7 também. Escolhe-se os representantes CLYPPT_sl2 e CLYPPT_sl6.

    correlation = soil_data[
        ["CRFVOL_sl1", "CRFVOL_sl2", "CRFVOL_sl3", "CRFVOL_sl4", "CRFVOL_sl5", "CRFVOL_sl6", "CRFVOL_sl7"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()
    # Todos os atributos CRFVOL têm alta correlação entre si. O CRFVOL_sl4 é uma boa escolha (ponto central)

    correlation = soil_data[["OCSTHA_sd1", "OCSTHA_sd2", "OCSTHA_sd3", "OCSTHA_sd4", "OCSTHA_sd5", "OCSTHA_sd6"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()
    # Todos os atributos OCSTHA têm alta correlação entre si (exceto talvez o sd1, que tem correlação por volta de
    # 0.6). Mesmo assim, deve ser possível escolher apenas o OCSTHA_sd4.

    correlation = soil_data[
        ["ORCDRC_sl1", "ORCDRC_sl2", "ORCDRC_sl3", "ORCDRC_sl4", "ORCDRC_sl5", "ORCDRC_sl6", "ORCDRC_sl7"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()
    # Verifica-se correlação entre os sl4 a sl7, e entre o sl2 e sl3. Reduz-se os dados para ORCDRC_sl1, ORCDRC_sl3 e
    # ORCDRC_sl6

    correlation = soil_data[
        ["PHIHOX_sl1", "PHIHOX_sl2", "PHIHOX_sl3", "PHIHOX_sl4", "PHIHOX_sl5", "PHIHOX_sl6", "PHIHOX_sl7"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()
    # Alta correlação no geral. Escolhe-se apenas o PHIHOX_sl4.

    correlation = soil_data[
        ["PHIKCL_sl1", "PHIKCL_sl2", "PHIKCL_sl3", "PHIKCL_sl4", "PHIKCL_sl5", "PHIKCL_sl6", "PHIKCL_sl7"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()
    # Alta correlação no geral. Escolhe-se apenas o PHIKCL_sl4.

    correlation = soil_data[
        ["SLTPPT_sl1", "SLTPPT_sl2", "SLTPPT_sl3", "SLTPPT_sl4", "SLTPPT_sl5", "SLTPPT_sl6", "SLTPPT_sl7"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()
    # Correlação altíssima entre todos os atributos. Escolhe-se apenas o SLTPPT_sl4.

    correlation = soil_data[
        ["SNDPPT_sl1", "SNDPPT_sl2", "SNDPPT_sl3", "SNDPPT_sl4", "SNDPPT_sl5", "SNDPPT_sl6", "SNDPPT_sl7"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()
    # Correlação altíssima entre todos os atributos. Escolhe-se apenas o SNDPPT_sl4.

    # O atributo BDRICM_BDRICM_M é igual para todos os fields. Logo, não tem variação para ser calculada a correlação.
    correlation = soil_data[["BDRLOG_BDRLOG_M", "BDTICM_BDTICM_M"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()
    # Esses atributos não têm correlação.

    # Como esses são dados estáticos, deve-se compará-los à média da produção de seus respectivos fields, para ver se
    # as pequenas diferenças de solo entre os fields podem ter algum efeito na produção.
    # Temos a média das produções por field já computada no vetor prod_mean_by_field.
    soil_data_with_prod = soil_data.merge(pd.DataFrame({'field': field_index, 'production_mean': prod_mean_by_field}),
                                          how="inner", left_on=["field"], right_on=["field"])
    correlation = soil_data_with_prod[
        ["BDRLOG_BDRLOG_M", "BDTICM_BDTICM_M", "BLDFIE_sl1", "BLDFIE_sl3", "BLDFIE_sl7", "CECSOL_sl4", "CLYPPT_sl2",
         "CLYPPT_sl6", "CRFVOL_sl4", "OCSTHA_sd4", "ORCDRC_sl1", "ORCDRC_sl3", "ORCDRC_sl6", "PHIHOX_sl4", "PHIKCL_sl4",
         "SLTPPT_sl4", "SNDPPT_sl4", "production_mean"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()
    # Mesmo reduzindo bastante o número de atributos, esse gráfico ainda tem uma visualização bem prejudicada.
    # Não é possível observar muita correlação entre os dados de solo diferentes (com exceção dos OCSTHA e os ORCDRC).
    # Alguns pontos de correlação negativa podem ser observados.
    # No entanto, NENHUM dos atributos de solo tem correlação significativa com a média da produção por field. Por isso,
    # é possível concluir que:
    #  - ou os atributos de solo são irrelevantes para a produção,
    #  - ou os fields são tão parecidos entre si em termos de solo que não é possível observar variações na produção
    # causadas por eles.
    # Os atributos mais significativos para a produção (embora pouco significativos) foram:
    #  - BLDFIE_sl7      (-0.35)
    #  - CRFVOL_sl4      ( 0.2)
    #  - ORCDRC_sl6      (-0.2)
    #  - BDRLOG_BDRLOG_M ( 0.19)
    #  - CECSOL_sl4      (-0.19)
    #  - CLYPPT_sl6      ( 0.18)

    full_dataset = pd.read_csv("monolith.csv")

    # --- Cálculo da correlação entre atributos de umidade ---
    correlation = full_dataset[["Soilwater_L1", "Soilwater_L2", "Soilwater_L3", "Soilwater_L4", "dewpoint", "production"]].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()
    # Todos os Soilwater e o dewpoint têm bastante correlação entre si. Pode-se escolher o Soilwater_L2 como o
    # representante deles. Aliás, o Soilwater_L4 tem ligeiramente menos correlação.
    # Parece que possuem um efeito considerável na produção (mas não linear).
    # O interessante é que o valor da correlação foi negativo: menos umidade = mais produção?

    features_to_compute = ["age", "temperature", "windspeed", "Precipitation", "acc_precipitation", "Soilwater_L2",
                           "Soilwater_L4", "production"]

    correlation = full_dataset[features_to_compute].corr()
    sns.heatmap(correlation, annot=True, linewidths=.5, cmap="Blues", vmax=1.0, vmin=-1.0)
    plt.show()

    # TODOS OS ATRIBUTOS (PARA COPIAR E COLAR)
    # Entradas de treinamento:
    # "field", "age", "type"
    #
    # Dados do campo:
    # "temperature", "windspeed", "Precipitation"
    # "Soilwater_L1", "Soilwater_L2", "Soilwater_L3", "Soilwater_L4", "dewpoint"
    #
    # Solo:
    # "BDRICM_BDRICM_M", "BDRLOG_BDRLOG_M", "BDTICM_BDTICM_M",
    # "BLDFIE_sl1", "BLDFIE_sl2", "BLDFIE_sl3", "BLDFIE_sl4", "BLDFIE_sl5", "BLDFIE_sl6", "BLDFIE_sl7"
    # "CECSOL_sl1", "CECSOL_sl2", "CECSOL_sl3", "CECSOL_sl4", "CECSOL_sl5", "CECSOL_sl6", "CECSOL_sl7"
    # "CLYPPT_sl1", "CLYPPT_sl2", "CLYPPT_sl3", "CLYPPT_sl4", "CLYPPT_sl5", "CLYPPT_sl6", "CLYPPT_sl7"
    # "CRFVOL_sl1", "CRFVOL_sl2", "CRFVOL_sl3", "CRFVOL_sl4", "CRFVOL_sl5", "CRFVOL_sl6", "CRFVOL_sl7"
    # "OCSTHA_sd1", "OCSTHA_sd2", "OCSTHA_sd3", "OCSTHA_sd4", "OCSTHA_sd5", "OCSTHA_sd6"
    # "ORCDRC_sl1", "ORCDRC_sl2", "ORCDRC_sl3", "ORCDRC_sl4", "ORCDRC_sl5", "ORCDRC_sl6", "ORCDRC_sl7"
    # "PHIHOX_sl1", "PHIHOX_sl2", "PHIHOX_sl3", "PHIHOX_sl4", "PHIHOX_sl5", "PHIHOX_sl6", "PHIHOX_sl7"
    # "PHIKCL_sl1", "PHIKCL_sl2", "PHIKCL_sl3", "PHIKCL_sl4", "PHIKCL_sl5", "PHIKCL_sl6", "PHIKCL_sl7"
    # "SLTPPT_sl1", "SLTPPT_sl2", "SLTPPT_sl3", "SLTPPT_sl4", "SLTPPT_sl5", "SLTPPT_sl6", "SLTPPT_sl7"
    # "SNDPPT_sl1", "SNDPPT_sl2", "SNDPPT_sl3", "SNDPPT_sl4", "SNDPPT_sl5", "SNDPPT_sl6", "SNDPPT_sl7"

if __name__ == "__main__":
    main()

# ==============================================================================
