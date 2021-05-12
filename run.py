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

df = pd.read_csv("./test_data/babybonusTest.csv")

testData = []
global adoptionFAQ
global babybonusFAQ
global comcareFAQ
global covid19FAQ
faqStore = './faqStore_test'

for i in range(len(df)):
    original = df.loc[i, "original"]
    re = df.loc[i, "reframed"]
    testData.append([original, re])


def csvReader(path):
    """
    the csv is assumed to have questions on column 1 and answers in column 2
    WITH NO HEADER
    """

    df = pd.read_csv(path, header=None)
    questions = df.iloc[:, 0]
    answers = df.iloc[:, 1]

    return questions, answers


def makeFAQ():
    global faqStore
    names = ["SymSub", "FPM", "EDA", "nlpAug"]
    quantity = [3, 3, 3, 2]

    generatorManager = GenerateManager(
        producers=[
            SymSubGenerator(dummyEN("lite")),
            FPMGenerator(),
            EDAGenerator(),
            AUG(),
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

    adoptionFAQ.save(faqStore)
    babybonusFAQ.save(faqStore)
    comcareFAQ.save(faqStore)
    covid19FAQ.save(faqStore)


def loadFAQ():
    global adoptionFAQ
    global babybonusFAQ
    global comcareFAQ
    global covid19FAQ
    global faqStore
    
    adoptionFAQ = FAQ(name="Adoption")
    babybonusFAQ = FAQ(name="Baby Bonus")
    comcareFAQ = FAQ(name="ComCare")
    covid19FAQ = FAQ(name="Covid 19")

    adoptionFAQ.load(faqStore)
    babybonusFAQ.load(faqStore)
    comcareFAQ.load(faqStore)
    covid19FAQ.load(faqStore)


# makeFAQ()
loadFAQ()

bot = Bani(
    FAQs=[adoptionFAQ, babybonusFAQ, comcareFAQ, covid19FAQ],
    modelPath="./latest_model3",
)


# "batchHardTriplet", "contrastiveLoss", "tripletLoss", "softmaxLayerLoss"
# bot.train("./", epochs=5, batchSize=64, lossName="batchHardTriplet")

# bot.saveModel("./latest_model3")


acc = bot.test(1, testData, i)
print(acc)

ans = bot.findClosest("how is the weather today")

for i in range(4):
    print(ans[i].maxScore)
    print(ans[i].score)
