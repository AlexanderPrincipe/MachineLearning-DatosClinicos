from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, Normalizer
from pyspark.ml.feature import VectorIndexer, ChiSqSelector		
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# VARIABLES

EDAD = 0
GENERO = 2
ETNIA = 3
ZONA = 4
ESCOLARIDAD = 7
FUMADOR = 10
DIABETES = 11
HAS = 12 # hipertencion arterial sistemica
HTADM = 13
GLICEMIA = 14
ENF_CORONARIA = 16
T_SISTOLICA = 17
T_DIASTOLICA = 18
COLESTEROL_TOTAL = 20
TRIGLICERIDOS = 22
RCV_GLOBAL = 25
GLICEMIA_AYUNO = 29
PERIMETRO_ABDOMINAL = 31
PESO = 33
TALLA = 34
IMC = 35
CREATININA = 37
MICROALBUMINURIA = 39
ESTADO_IRC = 42
FARMACOS_ANTIHIPERTENSIVOS = 44

# Convertir etnias
def etnia(p):
    if p == 'Afrocolombiano':
        return 1 
    elif p == 'Blanco':
        return 2
    elif p == 'Indigena':
        return 3
    elif p == 'Mestizo':
        return 4
    else:
        return 5

# Convertir zonas
def zona(p):
    if p == 'Rural':
        return 1 
    elif p == 'Urbana':
        return 2
    else:
        return 3


# Convertir fumador
def fumador(p):
    if p == 'Si':
        return 1 
    elif p == 'No':
        return 2
    else:
        return 3

# Convertir diabetes
def diabetes(p):
    if p == 'Si':
        return 1 
    elif p == 'No':
        return 2
    else:
        return 3

# Convertir HAS
def has(p):
    if p == 'Si':
        return 1 
    elif p == 'No':
        return 2
    else:
        return 3

# Convertir HTADM
def htadm(p):
    if p == 'Si':
        return 1 
    elif p == 'No':
        return 2
    else:
        return 3

# Convertir enfermedad coronaria
def enf_coronaria(p):
    if p == 'Si':
        return 1 
    elif p == 'No':
        return 2
    else:
        return 3

# Convertir RCV GLobal
def rcv(p):
    if p == 'ALTO':
        return 1 
    elif p == 'INTERMEDIO':
        return 2
    elif p == 'LATENTE':
        return 3
    elif p == 'MUY ALTO':
        return 4
    else:
        return 5

# Convertir Estado IRC
def irc(p):
    if p == 'ERC LEVE':
        return 1 
    elif p == 'ERC MODERADA':
        return 2
    elif p == 'ERC SEVERA':
        return 3
    elif p == 'NORMAL':
        return 4
    elif p == 'RIÑON TERMINAL':
        return 5
    else:
        return 6

# Cargar el CSV
def leer_df():
	conf = SparkConf().setAppName("TrabajoFinal").setMaster("local")
	sc = SparkContext(conf=conf)

	sqlContext = SQLContext(sc)
	
	# Creacion de rdd
	rdd = sqlContext.read.csv("BD.csv", header=True).rdd

	# Filtrando datos vacios
	rdd = rdd.filter(
		lambda x: (		   x[DIABETES] != None and x[DIABETES] != '' and x[EDAD] != None and x[GENERO] != None and x[ETNIA] != None and \
						   x[ZONA] != None and x[ESCOLARIDAD] != None and x[FUMADOR] != None and x[HAS] != None and \
						   x[HTADM] != None and x[GLICEMIA] != None and x[ENF_CORONARIA] != None and x[T_SISTOLICA] != None and x[T_DIASTOLICA] != None and \
						   x[COLESTEROL_TOTAL] != None and x[TRIGLICERIDOS] != None and x[RCV_GLOBAL] != None and x[GLICEMIA_AYUNO] != None and x[PERIMETRO_ABDOMINAL] != None and \
						   x[PESO] != None and x[TALLA] != None and x[IMC] != None and x[CREATININA] != None and x[MICROALBUMINURIA] != None and x[ESTADO_IRC] != None and \
						   x[FARMACOS_ANTIHIPERTENSIVOS] != None))

	# Consideramos los principales features
	rdd = rdd.map(
		lambda x: ( int(x[EDAD]), int(x[GENERO]), int(etnia(x[ETNIA])), int(zona(x[ZONA])), int(x[ESCOLARIDAD]), 
					int(fumador(x[FUMADOR])) , int(diabetes(x[DIABETES])), int(has(x[HAS])), int(htadm(x[HTADM])), int(x[GLICEMIA]),
					int(enf_coronaria(x[ENF_CORONARIA])), int(x[T_SISTOLICA]), int(x[T_DIASTOLICA]) , int(x[COLESTEROL_TOTAL]), float(x[TRIGLICERIDOS]),
				    int(rcv(x[RCV_GLOBAL])), int(x[GLICEMIA_AYUNO]), int(x[PERIMETRO_ABDOMINAL]), int(x[PESO]), int(x[TALLA]),
				    int(x[IMC]), int(x[CREATININA]), float(x[MICROALBUMINURIA]), int(irc(x[ESTADO_IRC])), int(x[FARMACOS_ANTIHIPERTENSIVOS]) ))
	df = rdd.toDF(["EDAD","GENERO","ETNIA","ZONA","ESCOLARIDAD",
	 			   "FUMADOR", "DIABETES","HAS","HTADM","GLICEMIA",
				   "ENF_CORONARIA","T_SISTOLICA","T_DIASTOLICA","COLESTEROL_TOTAL","TRIGLICERIDOS",
				   "RCV_GLOBAL","GLICEMIA_AYUNO","PERIMETRO_ABDOMINAL","PESO","TALLA",
				   "IMC","CREATININA","MICROALBUMINURIA","ESTADO_IRC","FARMACOS_ANTIHIPERTENSIVOS"])

	return df


# Analizar con los features mas representativos para el modelo
def leer_df_categoricos():
	conf = SparkConf().setAppName("Tarea4").setMaster("local")
	sc = SparkContext(conf=conf)

	sqlContext = SQLContext(sc)
	
	# Creacion de rdd
	rdd = sqlContext.read.csv("BD.csv", header=True).rdd
	
	# Filtramos los datos vacios
	rdd = rdd.filter(
		lambda x: (		   x[21] != None and x[21] != '' and x[55] != None and x[57] != None and x[63] != None and \
						   x[71] != None and x[82] != None and x[87] != None and x[54] != None and x[66] != None and x[59] != None and x[65] != None  ))

	# Features mas representativos
	rdd = rdd.map(
        lambda x: ( int(x[EDAD]), int(x[GENERO]), int(etnia(x[ETNIA])), int(x[GLICEMIA]), int(x[PERIMETRO_ABDOMINAL]), 
		            int(rcv(x[RCV_GLOBAL])), int(x[IMC]), int(diabetes(x[DIABETES])) ))
	df = rdd.toDF(["EDAD", "GENERO", "ETNIA", "GLICEMIA", "PERIMETRO_ABDOMINAL", "RCV_GLOBAL", "IMC", "DIABETES"])

	return df

# Seleccionar los features mas representativos para el modelo
def feature_selection(df):
	# Creamos vectorassembler
	assembler = VectorAssembler(
		inputCols=["EDAD","GENERO","ETNIA","ZONA","ESCOLARIDAD",
	 			   "FUMADOR","HAS","HTADM","GLICEMIA",
				   "ENF_CORONARIA","T_SISTOLICA","T_DIASTOLICA","COLESTEROL_TOTAL","TRIGLICERIDOS",
				   "RCV_GLOBAL","GLICEMIA_AYUNO","PERIMETRO_ABDOMINAL","PESO","TALLA",
				   "IMC","CREATININA","MICROALBUMINURIA","ESTADO_IRC","FARMACOS_ANTIHIPERTENSIVOS"],
		outputCol="features")
	df = assembler.transform(df)

	# Vectorindexer   
	indexer = VectorIndexer(
		inputCol="features", 
		outputCol="indexedFeatures")
	
	df = indexer.fit(df).transform(df)

	# Prueba ChiSquare
	selector = ChiSqSelector(
		numTopFeatures=8,
		featuresCol="indexedFeatures",
		labelCol="DIABETES",
		outputCol="selectedFeatures")
	resultado = selector.fit(df).transform(df)
	resultado.select("features", "selectedFeatures").show()

def entrenamiento(df):
	# Vectorizo
	df = df.select("EDAD", "GENERO", "ETNIA", "GLICEMIA", "PERIMETRO_ABDOMINAL", "RCV_GLOBAL", "IMC", "DIABETES")
	assembler = VectorAssembler(
		inputCols=["EDAD", "GENERO", "ETNIA", "GLICEMIA", "PERIMETRO_ABDOMINAL", "RCV_GLOBAL", "IMC"],
		outputCol="features")
	df = assembler.transform(df)

	# Dividir nuestro dataset
	(training_df, test_df) = df.randomSplit([0.7, 0.3])

	# Entrenamiento
	entrenador = DecisionTreeClassifier(
		labelCol="DIABETES", 
		featuresCol="features")

	# Creacion de pipeline
	pipeline = Pipeline(stages=[entrenador])
    # Se entrena el modelo
	model = pipeline.fit(training_df)

	# Prediccion
	predictions_df = model.transform(test_df)

	# Evaluador --> Accuracy
	evaluator = MulticlassClassificationEvaluator(
		labelCol="DIABETES",
		predictionCol="prediction",
		metricName="accuracy")

	# Exactitud
	exactitud = evaluator.evaluate(predictions_df)
	print("Exactitud: {}".format(exactitud))

def main():
	df = leer_df()
	#df = leer_df_categoricos()
	feature_selection(df)
	#entrenamiento(df)
	
if __name__ == "__main__":
	main()
