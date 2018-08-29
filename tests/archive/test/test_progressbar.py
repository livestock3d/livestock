import progressbar
import time
import datetime


def format_custom_text():
    format_custom_text = progressbar.FormatCustomText(
        'Spam: %(spam).1f kg, eggs: %(eggs)d',
        dict(
            spam=0.25,
            eggs=3,
        ),
    )

    bar = progressbar.ProgressBar(widgets=[
        format_custom_text,
        ' :: ',
        progressbar.Percentage(),
    ])
    for i in bar(range(25)):
        format_custom_text.update_mapping(eggs=i * 2)
        time.sleep(0.1)


def format_label():
    widgets = [progressbar.FormatLabel(
        'Processed: %(value)d lines (in: %(elapsed)s)')]
    bar = progressbar.ProgressBar(widgets=widgets)
    for i in bar((i for i in range(15))):
        time.sleep(0.1)


def print_output():
    for i in progressbar.progressbar(range(100), redirect_stdout=True):
        print('Some text', i)
        time.sleep(0.1)

def cmf_bar():

    widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' [', progressbar.AdaptiveETA(), ' ]']
    bar = progressbar.ProgressBar(max_value=20, widgets=widgets)
    for i in range(20):
        time.sleep(0.1)
        bar.update()


#format_custom_text()
cmf_bar()
#format_label()
#print_output()