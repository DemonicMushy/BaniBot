import pandas as pd
from typing import List, Tuple, Dict

from Bani.Bani import Bani
from Bani.core.FAQ import FAQ
from Bani.core.generation import GenerateManager
from Bani.core.defaults import defaultGenerateManager

from Bani.generation.rajat_work.qgen.generator.symsub import SymSubGenerator
from Bani.generation.rajat_work.qgen.generator.fpm.fpm import FPMGenerator
from Bani.generation.rajat_work.qgen.encoder.dummy import dummyEN
from Bani.generation.rajat_work.qgen.generator.eda import EDAGenerator
from Bani.generation.sentAug.sentAug import AUG
from Bani.generation.t5_paraphrase_gen.t5_paraphrase import T5Generator

from datetime import datetime


def csvReader(path):
    """
    the csv is assumed to have questions on column 1 and answers in column 2
    WITH NO HEADER
    """

    df = pd.read_csv(path, header=None)
    questions = df.iloc[:, 0]
    answers = df.iloc[:, 1]

    return questions, answers


def makeFAQ(targetDir_faqStore):
    """
    Currently, takes in data hardcoded within this function. (Adoption, Babybonus, Comcare, Covid19)
    Generates permutations using Bani, and saves the faqStore to targetDir_faqStore
    """
    names = ["SymSub", "FPM", "EDA", "nlpAug", "T5"]
    quantity = [3, 3, 3, 2, 3]

    generatorManager = GenerateManager(
        producers=[
            SymSubGenerator(dummyEN("lite")),
            FPMGenerator(),
            EDAGenerator(),
            AUG(),
            T5Generator(),
        ],
        names=names,
        nums=quantity,
    )

    questions1, answers1 = csvReader("./data/adoption/adoption-copy.csv")
    questions2, answers2 = csvReader("./data/babybonus/babybonus-copy.csv")
    questions3, answers3 = csvReader("./data/comcare/comcare-copy.csv")
    questions4, answers4 = csvReader("./data/covid19/covid19-copy.csv")

    adoptionFAQ = FAQ(name="Adoption", questions=questions1, answers=answers1)
    babybonusFAQ = FAQ(name="Baby Bonus", questions=questions2, answers=answers2)
    comcareFAQ = FAQ(name="ComCare", questions=questions3, answers=answers3)
    covid19FAQ = FAQ(name="Covid 19", questions=questions4, answers=answers4)

    adoptionFAQ.buildFAQ(generatorManager)
    babybonusFAQ.buildFAQ(generatorManager)
    comcareFAQ.buildFAQ(generatorManager)
    covid19FAQ.buildFAQ(generatorManager)

    adoptionFAQ.save(targetDir_faqStore)
    babybonusFAQ.save(targetDir_faqStore)
    comcareFAQ.save(targetDir_faqStore)
    covid19FAQ.save(targetDir_faqStore)


def loadFAQ(targetDir_faqStore):
    """
    Takes in directory of faqStore
    Currently, hardcoded to use the 4 topics (Adoption, BabyBonus, ComCare, Covid19)
    Creates FAQ objects, loads data into FAQ objects from the targetDir_faqStore
    Returns the list of FAQ objects
    """

    adoptionFAQ = FAQ(name="Adoption")
    babybonusFAQ = FAQ(name="Baby Bonus")
    comcareFAQ = FAQ(name="ComCare")
    covid19FAQ = FAQ(name="Covid 19")

    adoptionFAQ.load(targetDir_faqStore)
    babybonusFAQ.load(targetDir_faqStore)
    comcareFAQ.load(targetDir_faqStore)
    covid19FAQ.load(targetDir_faqStore)

    return [adoptionFAQ, babybonusFAQ, comcareFAQ, covid19FAQ]


def trainModel(FAQ_list, targetDir_model):
    """
    Takes in list of FAQ objects and directory to where model is to be stored.
    Trains the model and saves it to the target directory.
    """
    bot = Bani(FAQs=FAQ_list, modelPath=None)
    
    starttime = datetime.now()
    current_time = starttime.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    # "batchHardTriplet", "contrastiveLoss", "tripletLoss", "softmaxLayerLoss"
    bot.train(targetDir_model, epochs=5, batchSize=64, lossName="batchHardTriplet")
    bot.saveModel(targetDir_model)

    print("Model has been saved to ", targetDir_model)

    endtime = datetime.now()
    current_time = endtime.strftime("%H:%M:%S")
    print("End Time =", current_time)
    duration = endtime - starttime
    print("Duration =", duration)



########## Filenames that need to be changed as needed

faqStore = "./test_faqStore"
modelPath = "./test_model"

##########

########## To make faqStore from question answer csv data ((un)comment as necessary)

makeFAQ(faqStore)

########## To train the model ((un)comment as necessary)

loaded_FAQs = loadFAQ(faqStore)
trainModel(loaded_FAQs, modelPath)

##########

########## To load Bani with FAQStore and trained model (to be run after making FAQ, and after training model)

loaded_FAQs = loadFAQ(faqStore)

bot = Bani(
    FAQs=loaded_FAQs,
    modelPath=modelPath,
)

##########

########## Testing model with test data ((un)comment as necessary) (Be sure to load Bani first)

# read Test data from csv to determine accuracy later

df = pd.read_csv("./test_data/babybonusTest.csv")
testData = []

for i in range(len(df)):
    original = df.loc[i, "original"]
    re = df.loc[i, "reframed"]
    testData.append([original, re])

acc = bot.test(1, testData, i)
print("Accuracy using Bani's test method:", acc)

print("")

print("Answering: 'How is the weather today'")
ans = bot.findClosest("how is the weather today")

for i in range(4):
    print("FAQ Name:\t", ans[i].faqName)
    print("FAQ ID:\t\t", ans[i].faqId)
    print("Max score:\t", ans[i].maxScore)
    print("Score:\t\t", ans[i].score)

##########