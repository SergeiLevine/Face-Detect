#!/bin/bash
clear
var=0
while [[ $var -ne 5 ]]; do
#clear
echo "Press 1 to extract aligned Images"
echo "Press 2 to generate embeddings"
echo "Press 3 to train classifier"
echo "Press 4 to execute face recognition" 
echo "Press 5 to to Exit" 
read var
if (($var == 5));then
echo "EXIT"
exit
fi

if (($var == 1));then
./util/align-dlib.py ./training-images/ align outerEyesAndNose ./aligned-images/ --size 96
fi

if (($var == 2));then
./batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/
fi

if (($var == 3));then
./util/classifier.py train ./generated-embeddings/
fi

if (($var == 4));then
./util/classifier.py infer generated-embeddings/classifier.pkl find-people/ --multi
fi


done
