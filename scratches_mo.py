from lxml import objectify
import xml.etree.cElementTree as ET
import untangle

    # TODO: read the solution template file and use their structure for the solution class
    # figure out how to do this from their case or package
solutionTemplateFile = r'log_example.xml'


with open(solutionTemplateFile, 'r') as f:
    xmlTemplate = f.read()

solclass = untangle.parse(xmlTemplate)





tree = ET.parse(solutionTemplateFile)
root = tree.getroot()

main = objectify.fromstring(xmlTemplate)
print(main.object1[0])  # content
print(main.object1[1])  # contenbar
print(main.object1[0].get("attr"))  # name
print(main.test)  # me



