from re import match
from music21 import converter, analysis, environment 
from enum import Enum, unique
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.machar import MachAr
import re


class MArray:
    mlist = None

    def __init__(self, item):
        self.mlist = item

    def __getitem__(self, key):
        return self.mlist[key % len(self.mlist)]

    def getIndex(self, item):
        return self.mlist.index(item)

    def getItemByDistance(self, item, distance):
        return self.__getitem__(self.getIndex(item)+distance)


s = converter.parse('Corpus/Etudes_and_Preludes/Bach,_Johann_Sebastian/The_Well-Tempered_Clavier_I/17/analysis_on_score.mxl')
bool_array = [0, 1, 0]


@unique
class Mode(Enum):
    Ionian = {"fifth": 7, "negativeModeName": "Aeolian", "adapter": MArray([0, 0, 0, -1, -1, 0, 0, 0, -1, -1, -1, -1])}
    Dorian = {"fifth": 7, "negativeModeName": "Mixolydian", "adapter": MArray([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0])}
    Phrygian = {"fifth": 7, "negativeModeName": "Lydian", "adapter": MArray([0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1])}
    Lydian = {"fifth": 7, "negativeModeName": "Phrygian", "adapter": MArray([0, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1])}
    Mixolydian = {"fifth": 7, "negativeModeName": "Dorian", "adapter": MArray([0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0])}
    Aeolian = {"fifth": 7, "negativeModeName": "Ionian", "adapter": MArray([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1])}
    Locrian = {"fifth": 6, "negativeModeName": "Locrian", "adapter": MArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
    Hexatonic = {"fifth": 6, "negativeModeName": "Hexatonic", "adapter": MArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
    Diminished = {"fifth": 6, "negativeModeName": "Diminished", "adapter": MArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
    Augmented = {"fifth": 8, "negativeModeName": "Augmented", "adapter": MArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
    Chromatic = {"fifth": 7, "negativeModeName": "Chromatic", "adapter": MArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}


colors = MArray(["red", "green", "blue"])
key_names = MArray(["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"])
rome_names = MArray(["i", "bii", "ii", "biii", "iii", "iv", "bv", "v", "bvi", "vi", "bvii", "vii"])
text_adapter_dictionary = {"c": "C", "C": "C",
                            "db": "Db", "D-": "Db", "d-": "Db", "c#": "Db", "C#": "Db", "Db": "Db",
                            "d": "D", "D": "D",
                            "eb": "Eb", "E-": "Eb", "e-": "Eb", "d#": "Eb", "D#": "Eb", "Eb": "Eb",
                            "e": "E", "E": "E",
                            "f": "F", "F": "F",
                            "gb": "Gb", "G-": "Gb", "g-": "Gb", "f#": "Gb", "F#": "Gb", "Gb": "Gb",
                            "g": "G", "G": "G",
                            "ab": "Ab", "A-": "Ab", "a-": "Ab", "g#": "Ab", "G#": "Ab", "Ab": "Ab",
                            "a": "A", "A": "A",
                            "bb": "Bb", "B-": "Bb", "b-": "Bb", "a#": "Bb", "A#": "Bb", "Bb": "Bb",
                            "b": "B", "B": "B",
                            "I": "i", "i": "i",
                            "bII": "bii", "#i": "bii", "#I": "bii", "bii": "bii",
                            "II": "ii", "ii": "ii",
                            "bIII": "biii", "#ii": "biii", "#II": "biii", "biii": "biii",
                            "III": "iii", "iii": "iii",
                            "IV": "iv", "iv": "iv",
                            "bV": "bv", "#iv": "bv", "#IV": "bv", "bv": "bv",
                            "V": "v", "v": "v",
                            "bVI": "bvi", "#v": "bvi", "#V": "bvi", "bvi": "bvi",
                            "VI": "vi", "vi": "vi",
                            "bVII": "bvii", "#vi": "bvii", "#VI": "bvii", "bvii": "bvii",
                            "VII": "vii", "vii": "vii",
                            "I+": "i", "io": "i",
                            "bII+": "bii", "#io": "bii", "#I+": "bii", "biio": "bii",
                            "II+": "ii", "iio": "ii",
                            "bIII+": "biii", "#iio": "biii", "#II+": "biii", "biiio": "biii",
                            "III+": "iii", "iiio": "iii",
                            "IV+": "iv", "ivo": "iv",
                            "bV+": "bv", "#ivo": "bv", "#IV+": "bv", "bvo": "bv",
                            "V+": "v", "vo": "v",
                            "bVI+": "bvi", "#vo": "bvi", "#V+": "bvi", "bvio": "bvi",
                            "VI+": "vi", "vio": "vi",
                            "bVII+": "bvii", "#vio": "bvii", "#VI+": "bvii", "bviio": "bvii",
                            "VII+": "vii", "viio": "vii",
                            "iø": "i",
                            "#iø": "bii", "biiø": "bii",
                            "iiø": "ii",
                            "#iiø": "biii", "biiiø": "biii",
                            "iiiø": "iii",
                            "ivø": "iv",
                            "#ivø": "bv", "bvø": "bv",
                            "vø": "v",
                            "#vø": "bvi", "bviø": "bvi",
                            "viø": "vi",
                            "#viø": "bvii", "bviiø": "bvii",
                            "viiø": "vii",
                            "#iiio": "iii",
                            "#iii": "iii",
                            "C-": "B",
                            }
rome_mode_dictionary = {
    "io": Mode.Diminished,
    "iø": Mode.Locrian,
    "i": Mode.Aeolian, ''''''
    "I": Mode.Ionian, ''''''
    "I+": Mode.Augmented,

    "#io": Mode.Diminished,
    "#iø": Mode.Locrian,
    "#i": Mode.Aeolian,
    "#I": Mode.Ionian,
    "#I+": Mode.Augmented,

    "biio": Mode.Diminished,
    "biiø": Mode.Locrian,
    "bii": Mode.Aeolian,
    "bII": Mode.Ionian,
    "bII+": Mode.Augmented,

    "iio": Mode.Diminished,
    "iiø": Mode.Locrian, ''''''
    "ii": Mode.Dorian, ''''''
    "II": Mode.Ionian,
    "II+": Mode.Augmented,

    "#iio": Mode.Diminished,
    "#iiø": Mode.Locrian,
    "#ii": Mode.Aeolian,
    "#II": Mode.Ionian,
    "#II+": Mode.Augmented,

    "biiio": Mode.Diminished,
    "biiiø": Mode.Locrian,
    "biii": Mode.Aeolian,
    "bIII": Mode.Ionian, ''''''
    "bIII+": Mode.Augmented,

    "iiio": Mode.Diminished,
    "iiiø": Mode.Locrian,
    "iii": Mode.Phrygian, ''''''
    "III": Mode.Ionian,
    "III+": Mode.Augmented,

    "ivo": Mode.Diminished,
    "ivø": Mode.Locrian,
    "iv": Mode.Dorian, ''''''
    "IV": Mode.Lydian, ''''''
    "IV+": Mode.Augmented,

    "#ivo": Mode.Diminished,
    "#ivø": Mode.Locrian,
    "#iv": Mode.Aeolian,
    "#IV": Mode.Ionian,
    "#IV+": Mode.Augmented,

    "bvo": Mode.Diminished,
    "bvø": Mode.Locrian,
    "bv": Mode.Aeolian,
    "bV": Mode.Ionian,
    "bV+": Mode.Augmented,

    "vo": Mode.Diminished,
    "vø": Mode.Locrian,
    "v": Mode.Phrygian, ''''''
    "V": Mode.Mixolydian, ''''''
    "V+": Mode.Augmented,

    "#vo": Mode.Diminished,
    "#vø": Mode.Locrian,
    "#v": Mode.Aeolian,
    "#V": Mode.Ionian,
    "#V+": Mode.Augmented,

    "bvio": Mode.Diminished,
    "bviø": Mode.Locrian,
    "bvi": Mode.Aeolian,
    "bVI": Mode.Lydian, ''''''
    "bVI+": Mode.Augmented,

    "vio": Mode.Diminished,
    "viø": Mode.Locrian,
    "vi": Mode.Aeolian, ''''''
    "VI": Mode.Ionian,
    "VI+": Mode.Augmented,

    "#vio": Mode.Diminished,
    "#viø": Mode.Locrian,
    "#vi": Mode.Aeolian,
    "#VI": Mode.Ionian,
    "#VI+": Mode.Augmented,

    "bviio": Mode.Diminished,
    "bviiø": Mode.Locrian,
    "bvii": Mode.Aeolian,
    "bVII": Mode.Mixolydian, ''''''
    "bVII+": Mode.Augmented,

    "viio": Mode.Diminished,
    "viiø": Mode.Locrian,
    "vii": Mode.Aeolian,
    "VII": Mode.Ionian,
    "VII+": Mode.Augmented,

    "#iiio": Mode.Diminished,
    "#iii": Mode.Phrygian
}

analyze_parts_dictionary = {}
ambitusAnalyzer = analysis.discrete.Ambitus()

# print(s.analyze('key'))
# s.remove(s.parts[2])
for part in s.recurse().parts:
    if part.id == 'Analysis':
        analyze_parts_dictionary[part.activeSite.id] = part

for score_id in analyze_parts_dictionary:
    analyze_part = analyze_parts_dictionary[score_id]
    score = analyze_part.activeSite
    pitchMin, pitchMax = ambitusAnalyzer.getPitchSpan(score)
    score.midiRangeCenter = pitchMin.midi + pitchMax.midi
    '''
    for fobj in score.flat.notes:
        if fobj.isNote:
            fobj.pitch.midi=score.midiRangeCenter - fobj.pitch.midi
        elif fobj.isChord:
            for pobj in fobj.pitches:
                pobj.midi = score.midiRangeCenter - pobj.midi 
    '''
    analyze_list = []
    key_record = None
    for note_ in analyze_part.recurse().notes:
        if note_.lyric is not None:
            # if note_.isChord:
            analyze_list.append([note_, note_.getOffsetBySite(score.flat)])
            lyric_ = note_.lyric
            lyric___ = note_.lyric
            note_.lyric = None
            char_index = lyric_.find(': ')
            neb_index = lyric_.find('/')
            neb = None
            if neb_index != -1:
                neb = lyric_[neb_index+1:]
                lyric_ = lyric_[:neb_index]
            if char_index != -1:
                note_.is_key_changed = True
                key_record = lyric_[:char_index]
                lyric_ = lyric_[char_index+2:]
            else:
                note_.is_key_changed = False

            match_ = re.search('\d+', lyric_)
            match_string = ''
            if match_ is not None:
                match_index = match_.start()
                while lyric_[match_index-1] == 'b' or lyric_[match_index-1] == '#':
                    match_index -= 1
                match_string = lyric_[match_index:]
                lyric_ = lyric_[:match_index]
            #note_.addLyric(lyric_)#
            #note_.addLyric(match_string)#
            if neb is not None:
                note_.addLyric(key_names.getItemByDistance(text_adapter_dictionary[key_names.getItemByDistance(
                    text_adapter_dictionary[key_record], rome_names.getIndex(text_adapter_dictionary[neb]))], rome_names.getIndex(text_adapter_dictionary[lyric_])))
            else:
                note_.addLyric(key_names.getItemByDistance(
                    text_adapter_dictionary[key_record], rome_names.getIndex(text_adapter_dictionary[lyric_])))
            note_.addLyric(rome_mode_dictionary[lyric_].name)
            note_.addLyric(lyric___)
            # note_.lyrics[0].text=note_.quality

    for i in range(len(analyze_list)):
        if i+1 < len(analyze_list):
            analyze_list[i].append(analyze_list[i+1][1])
        else:
            analyze_list[i].append(
                analyze_list[i][0].quarterLength + analyze_list[i][1])
    color_index = 0
    negative_axis = None
    if analyze_list is not None:
        positive_index_by_name = analyze_list[0][0].lyrics[0].text
        negative_index_by_name = key_names.getItemByDistance(
            positive_index_by_name, Mode[analyze_list[0][0].lyrics[1].text].value['fifth'])
        positive_index = key_names.getIndex(positive_index_by_name)
        negative_index = key_names.getIndex(negative_index_by_name)
        negative_axis = [negative_index_by_name,
                            positive_index_by_name, negative_index, positive_index]
        # print(negative_axis)

    for item in analyze_list:
        pitches_ = []
        # print('----')
        # .getElementsByOffset(item[1],item[2]):
        for note_ in score.recurse().notes:
            offsite___ = note_.getOffsetBySite(score.flat)
            if offsite___ >= item[1] and offsite___ < item[2]:
                note_.style.color = colors[color_index]
                if note_.isNote:
                    pitches_.append(note_.pitch)
                    # note_.pitch.midi+=1
                    # print(note_)
                elif note_.isChord:
                    for note___ in note_:
                        pitches_.append(note___.pitch)
                        # note___.pitch.midi+=1
        item.append(False)  # is V-I 3
        item.append(item[0].lyrics[1].text)  # ModeName 4
        item.append(key_names.getIndex(
            item[0].lyrics[0].text))  # positiveIndex 5
        item.append(Mode[item[4]].value['negativeModeName'])
        item.append(negative_axis[2]-item[5] -
                    Mode[item[4]].value['fifth']+negative_axis[3])

        # print(item)
        #pitches_ = list(set(pitches_))
        if bool_array[0]:
            for pitch_ in pitches_:
                #midi_12= pitch_.midi%12
                # print('-======')
                # print(pitch_.midi)
                #pitch_.midi+= Mode[item[4]].value['adapter'][midi_12-item[5]]
                # if not hasattr(pitch_, 'hasBeenChanged'):
                # pitch_.hasBeenChanged=True
                midi_12 = pitch_.midi % 12
                pitch_.midi += Mode[item[4]].value['adapter'][midi_12-item[5]]
                move_ = (item[5]-item[7]) % 12
                if move_ < -5:
                    move_ += 12
                elif move_ > 5:
                    move_ -= 12

                pitch_.midi -= move_
                # pitch_.midi+=1
                #    None
                None
                # print(Mode[item[4]].value['adapter'].mlist)
                # print(midi_12)
                # print(pitch_.midi)
                #pitch_.midi-= (item[5]-item[7])%12
            item[0].lyric = None
            item[0].addLyric(key_names[item[7]])
            item[0].addLyric(item[6])
        elif bool_array[1]:
            is_change = False
            index__ = analyze_list.index(item)
            if index__ < len(analyze_list)-1:

                distance__ = key_names.getIndex(
                    item[0].lyrics[0].text) - key_names.getIndex(analyze_list[index__+1][0].lyrics[0].text)
                if distance__ == 7 or distance__ == -5:
                    item[3] = True
                    if index__+2 < len(analyze_list):
                        distance__next = key_names.getIndex(
                            analyze_list[index__+1][0].lyrics[0].text)-key_names.getIndex(analyze_list[index__+2][0].lyrics[0].text)
                        if distance__next != 7 and distance__next != -5:
                            is_change = True
                        #is_change = True
                    else:
                        is_change = True

                    for pitch_ in pitches_:
                        item[0].style.color = 'yellow'

            if is_change:
                # print(item)
                for pitch_ in pitches_:
                    midi_12 = pitch_.midi % 12
                    pitch_.midi += Mode[item[4]
                                        ].value['adapter'][midi_12-item[5]]
                    if bool_array[2]:
                        pitch_.midi -= 2
                
                item[0].lyric = None
                if bool_array[2]:
                    item[0].addLyric(key_names[item[5] - 2])
                else:
                    item[0].addLyric(key_names[item[5]])
                item[0].addLyric(item[6])
        color_index += 1
    # print(Mode['Ionian'].value['adapter'][101])

    # print(analyze_list)
environment.set('musicxmlPath', '/usr/bin/musescore3')
s.show()