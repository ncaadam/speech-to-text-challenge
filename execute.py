import sys
from classes import Preprocessor,Modeler

def main(input_path,output_path):
	preprocessor = Preprocessor(input_path)
	metadata_dataframe = preprocessor.preprocess_data()
	preprocessor.write_to_disk(output_path)

	modeler = Modeler(output_path)
	predictions = modeler.score_dataset()
	print(modeler.get_classification_report())

if __name__ == '__main__':
	main(sys.argv[1],sys.argv[2])
