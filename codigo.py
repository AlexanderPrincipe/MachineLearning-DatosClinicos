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

# Convertir a entero jugadores ofensivos y jugadores defensivos
def tipo_posiciones(p):
    # Ofensivo
    if p == 'RF' or p == 'ST' or p == 'LF' or p == 'RS' or p == 'LS' or p == 'CF' or p == 'LW' or p == 'RCM' or p == 'LCM' or p == 'LDM' or p == 'CAM' or p == 'CDM' or p == 'RM' or p == 'LAM' or p == 'LM' or p == 'RDM' or p == 'RW' or p == 'CM' or p == 'RAM':
        return 1 
    # Defensivo
    elif p == 'RCB' or p == 'CB' or p == 'LCB' or p == 'LB' or p == 'RB' or p == 'RWB' or p == 'LWB' or p == 'GK':
        return 0
    else:
        return 1

# Cargar el CSV
def leer_df():
	conf = SparkConf().setAppName("TrabajoFinal").setMaster("local")
	sc = SparkContext(conf=conf)

	sqlContext = SQLContext(sc)
	
	# Creacion de rdd
	rdd = sqlContext.read.csv("BDClinica.csv", header=True).rdd

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
		lambda x: ( int(x[EDAD]), int(x[GENERO]), str(x[ETNIA]),str(x[ZONA]), int(x[ESCOLARIDAD]), 
					str(x[FUMADOR]) , str(x[DIABETES]), str(x[HAS]), str(x[HTADM]), int(x[GLICEMIA]),
					str(x[ENF_CORONARIA]), int(x[T_SISTOLICA]), int(x[T_DIASTOLICA]) , int(x[COLESTEROL_TOTAL]), float(x[TRIGLICERIDOS]),
				    str(x[RCV_GLOBAL]), int(x[GLICEMIA_AYUNO]), int(x[PERIMETRO_ABDOMINAL]), int(x[PESO]), int(x[TALLA]),
				    int(x[IMC]), int(x[CREATININA]), float(x[MICROALBUMINURIA]), str(x[ESTADO_IRC]), int(x[FARMACOS_ANTIHIPERTENSIVOS]) ))
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
	rdd = sqlContext.read.csv("BDClinica.csv", header=True).rdd
	
	# Filtramos los datos vacios
	rdd = rdd.filter(
		lambda x: (		   x[21] != None and x[21] != '' and x[55] != None and x[57] != None and x[63] != None and \
						   x[71] != None and x[82] != None and x[87] != None and x[54] != None and x[66] != None and x[59] != None and x[65] != None  ))

	# Features mas representativos
	rdd = rdd.map(
        lambda x: (tipo_posiciones((x[21])) ,int(x[55].split('+')[0]), int(x[57].split('+')[0]), int(x[63].split('+')[0]),
		int(x[71].split('+')[0]),int(x[82].split('+')[0]), int(x[87].split('+')[0]),int(x[54].split('+')[0]),int(x[66].split('+')[0]),int(x[59].split('+')[0]), int(x[65].split('+')[0])  ))
	df = rdd.toDF(["Position", "Finishing", "ShortPassing", "BallControl", "Stamina", "SlidingTackle", "GKReflexes", "Crossing", "Agility", "Dribbling", "SprintSpeed"])

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
		numTopFeatures=10,
		featuresCol="indexedFeatures",
		labelCol="DIABETES",
		outputCol="selectedFeatures")
	resultado = selector.fit(df).transform(df)
	resultado.select("features", "selectedFeatures").show()

def entrenamiento(df):
	# Vectorizo
	df = df.select("Finishing", "ShortPassing", "BallControl", "Stamina", "SlidingTackle", "GKReflexes", "Crossing", "Agility", "Position", "Dribbling", "SprintSpeed")
	assembler = VectorAssembler(
		inputCols=["Finishing", "ShortPassing", "BallControl", "Stamina", "SlidingTackle", "GKReflexes", "Crossing", "Agility", "Dribbling", "SprintSpeed"],
		outputCol="features")
	df = assembler.transform(df)

	# Dividir nuestro dataset
	(training_df, test_df) = df.randomSplit([0.7, 0.3])

	# Entrenamiento
	entrenador = DecisionTreeClassifier(
		labelCol="Position", 
		featuresCol="features")

	# Creacion de pipeline
	pipeline = Pipeline(stages=[entrenador])
    # Se entrena el modelo
	model = pipeline.fit(training_df)

	# Prediccion
	predictions_df = model.transform(test_df)

	# Evaluador --> Accuracy
	evaluator = MulticlassClassificationEvaluator(
		labelCol="Position",
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
