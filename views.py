
#!/usr/bin/env python3
 
#Creating a dictionary with a view for each scene
views = {"01": {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 2.1708788571615756, 2.3478365775420635, 0.67437258194996286 ],
                    "boundingbox_min" : [ -0.56574843049847423, -0.46347421852038145, -0.3499609535302719 ],
                    "field_of_view" : 60.0,
                    "front" : [ -0.63222296093356012, -0.6562712791708788, 0.41182779872638609 ],
                    "lookat" : [ -0.068224155371651343, -0.51271966795867796, 0.039849172328959702 ],
                    "up" : [ 0.31890074619870667, 0.26400754903281515, 0.91027596262210175 ],
                    "zoom" : 0.38119999999999998
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[0.766, -0.643, 0, -0.03], [-0.22, -0.262, -0.94, 0.2], [0.604, 0.72, -0.342, 1.306], [0, 0, 0, 1]]
        },

        "02": {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 2.0273444330909034, 2.1076340040646029, 0.5 ],
                    "boundingbox_min" : [ -0.75107110106061392, -0.78540220153256346, -0.76186610291479739 ],
                    "field_of_view" : 60.0,
                    "front" : [ -0.67158622163039705, -0.66013321706764561, 0.33644625520300891 ],
                    "lookat" : [ 0.20455800372390379, -0.0036187189107478638, 0.021959775363446805 ],
                    "up" : [ 0.27775006196619934, 0.19667136451841946, 0.94030594885719798 ],
                    "zoom" : 0.42120000000000007
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[0.766, -0.643, 0, -0.03], [-0.22, -0.262, -0.94, 0.25], [0.604, 0.72, -0.342, 1.406], [0, 0, 0, 1]]
        },

        "03": {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 2.2213514474360596, 2.2095889377205644, 0.5 ],
                    "boundingbox_min" : [ -0.62957359811024971, -0.76621639650684759, -0.51917502633783563 ],
                    "field_of_view" : 60.0,
                    "front" : [ -0.74164253088670629, -0.61722454380706204, 0.2626789274040176 ],
                    "lookat" : [ 0.60569959076656354, -0.16782634841821531, 0.067543897871059674 ],
                    "up" : [ 0.17423577568784301, 0.20089907934739565, 0.96399245556582258 ],
                    "zoom" : 0.52120000000000011
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[0.766, -0.643, 0, -0.1], [-0.22, -0.262, -0.94, 0.18], [0.604, 0.72, -0.342, 1.3], [0, 0, 0, 1]]
        },

        "04":{
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 1.8368455348953399, 1.3689408913666876, 0.5 ],
                    "boundingbox_min" : [ -0.71855983115395861, -0.53228570088509652, -1.1411296567944125 ],
                    "field_of_view" : 60.0,
                    "front" : [ -0.58875270626508702, -0.70177352256869396, 0.40110369468139151 ],
                    "lookat" : [ 0.070313014779345928, 0.16925218363908273, -0.46383907731632573 ],
                    "up" : [ 0.33254114939579182, 0.24199598338084907, 0.91151211071826233 ],
                    "zoom" : 0.58120000000000016
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[0.766, -0.643, 0, -0.2], [-0.22, -0.262, -0.94, 0.3], [0.604, 0.72, -0.342, 1.8], [0, 0, 0, 1]]
        },

        "05": {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 1.5134528826810116, 1.3939788188221123, 0.5 ],
                    "boundingbox_min" : [ -0.97076910437580555, -0.97074615683868237, -1.2715091028327103 ],
                    "field_of_view" : 60.0,
                    "front" : [ -0.79825003811151418, -0.47239860316499316, 0.37367959053543387 ],
                    "lookat" : [ 1.5971755955324141, -1.2607132872237177, 0.10038847627720142 ],
                    "up" : [ 0.2308809374380118, 0.33302841309819509, 0.91421336065332315 ],
                    "zoom" : 0.88120000000000043
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[0.766, -0.643, 0, 0], [-0.22, -0.262, -0.94, 0.1], [0.604, 0.72, -0.342, 2], [0, 0, 0, 1]]
        },

        "06": {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 1.8808644720671954, 1.8096838339799146, 0.5 ],
                    "boundingbox_min" : [ -0.90051787592444377, -0.95085324404846694, -1.2022636399501976 ],
                    "field_of_view" : 60.0,
                    "front" : [ -0.92763901652699465, -0.36934009258314376, 0.055441419782553264 ],
                    "lookat" : [ 2.311534549126721, -0.77322361237539294, 1.3118493338313935 ],
                    "up" : [ 0.002780612694742543, 0.14161222248142102, 0.98991830301137207 ],
                    "zoom" : 1.2799999999999994
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[0.766, -0.643, 0, 0], [-0.22, -0.262, -0.94, 0.06], [0.604, 0.72, -0.342, 1.8], [0, 0, 0, 1]]
        },

        "07": {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 1.3217193218122518, 1.0628330263700081, 0.5 ],
                    "boundingbox_min" : [ -1.3353068238866534, -0.87059286581593509, -1.1742447723908804 ],
                    "field_of_view" : 60.0,
                    "front" : [ -0.48876089031730047, -0.81955999808027635, 0.29905551598808117 ],
                    "lookat" : [ 1.2723332125840143, -1.4425085421533674, 0.61578311278893716 ],
                    "up" : [ -0.079949279998118405, 0.38342372093726218, 0.92010562592041967 ],
                    "zoom" : 0.58120000000000016
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[0.996, 0, -0.087, 0], [-0.082, -0.343, -0.936, 0.15], [-0.03, 0.94, -0.34, 2], [0, 0, 0, 1]]
        },

        "08": {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 1.3214459430226426, 0.94674826777960397, 0.5 ],
                    "boundingbox_min" : [ -0.734361654793245, -0.82707846906911731, -1.1840812242309653 ],
                    "field_of_view" : 60.0,
                    "front" : [ -0.34773243922262687, -0.81205231653338072, 0.46867172511796801 ],
                    "lookat" : [ 1.6011246568343216, -0.67568604275025779, 0.17674205363680706 ],
                    "up" : [ -0.00092914849952860368, 0.50016470236455801, 0.86592979345420318 ],
                    "zoom" : 0.7412000000000003
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[0.996, 0, -0.087, 0], [-0.082, -0.342, -0.936, 0.2], [-0.03, 0.94, -0.340, 2.1], [0, 0, 0, 1]]
        },

        "09":{
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 1.4108335757255555, 1.8656315724196468, 0.5 ],
                    "boundingbox_min" : [ -1.5554066276550294, -0.59031696117248122, -0.7018619589581585 ],
                    "field_of_view" : 60.0,
                    "front" : [ 0.13273428024912706, -0.97869223962743601, 0.1566624107429086 ],
                    "lookat" : [ -0.65933737315229113, 0.65642732046814567, 0.82345749128862566 ],
                    "up" : [ 0.010919221473667064, 0.15949530814468596, 0.9871382969382968 ],
                    "zoom" : 0.68120000000000025
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[1, 0, 0, 0], [0, -0.423, -0.906, 0.08], [0, 0.906, -0.423, 1.5], [0, 0, 0, 1]]
        },

        "10": {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 1.4008630204200745, 1.1534927546063607, 0.5 ],
                    "boundingbox_min" : [ -1.5554066276550294, -0.58052799429572066, -0.66513857560225953 ],
                    "field_of_view" : 60.0,
                    "front" : [ 0.39856453644384138, -0.81722730422871892, 0.41627616255602118 ],
                    "lookat" : [ -0.56971301844088629, -0.50513163912932246, 0.0014034838486766095 ],
                    "up" : [ -0.12986479626469727, 0.3990240747582815, 0.90769759416582463 ],
                    "zoom" : 0.48120000000000013
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[1, 0, 0, 0], [0, -0.275, -0.96, 0.2], [0, 0.96, -0.275, 1.9], [0, 0, 0, 1]]
        },

        "11": {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 1.4108335757255555, 1.7747136949032682, 0.5 ],
                    "boundingbox_min" : [ -1.5323825550079346, -0.36236236653478737, -0.69276225652465895 ],
                    "field_of_view" : 60.0,
                    "front" : [ 0.085727424007823447, -0.98994040025951757, 0.11255582040485132 ],
                    "lookat" : [ -0.61857445376185571, 0.24354580501566964, 0.6471226302560823 ],
                    "up" : [ 0.010758095933493094, 0.11388487084165502, 0.99343570479687571 ],
                    "zoom" : 0.50120000000000009
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[1, 0, 0, 0], [0, -0.39, -0.92, 0.2], [0, 0.92, -0.39, 1.3], [0, 0, 0, 1]]
        },

        "12": {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 1.4108335757255555, 1.4805311367906322, 0.5 ],
                    "boundingbox_min" : [ -1.5554066276550294, -0.36358016250059677, -0.6787935807997636 ],
                    "field_of_view" : 60.0,
                    "front" : [ -0.063510822884602344, -0.98308260060272035, 0.17179923098987646 ],
                    "lookat" : [ 0.26553079047931177, 0.85523871279112429, 0.56053988457495119 ],
                    "up" : [ 0.036823877384590184, 0.16972108033898656, 0.9848039180176591 ],
                    "zoom" : 0.6212000000000002
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[1, 0, 0, 0], [0, -0.309, -0.951, 0.35], [0, 0.951, -0.309, 1.6], [0, 0, 0, 1]]
        },

        "13": {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 0.89068112333615623, 1.9093434668676925, 0.72723170070425924 ],
                    "boundingbox_min" : [ -1.4507157969474793, -0.53472168620113147, -0.73811007370267989 ],
                    "field_of_view" : 60.0,
                    "front" : [ 0.19068435026743652, -0.91602131443438728, 0.35290852931741828 ],
                    "lookat" : [ 0.049600145037683216, 1.1269636233659026, -0.080946653321393558 ],
                    "up" : [ -0.13722059372142703, 0.33110241600214119, 0.93356397679874314 ],
                    "zoom" : 0.88120000000000043
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[1, 0, 0, 0], [0, -0.309, -0.951, 0.15], [0, 0.951, -0.309, 1.4], [0, 0, 0, 1]]
        },

        "14": {
            "class_name" : "ViewTrajectory",
            "interval" : 29,
            "is_loop" : False,
            "trajectory" : 
            [
                {
                    "boundingbox_max" : [ 1.4034442170461019, 1.6968587525400245, 0.82729373618304125 ],
                    "boundingbox_min" : [ -0.9955561316013336, -0.43800689901421774, -0.22993015344454504 ],
                    "field_of_view" : 60.0,
                    "front" : [ 0.36827486210190807, -0.90757368014089745, 0.20170186181423483 ],
                    "lookat" : [ -1.0125343134662794, -0.47074883000814866, 0.55820270960489971 ],
                    "up" : [ -0.041648643334803261, 0.20062722970446437, 0.97878194977711075 ],
                    "zoom" : 0.50120000000000009
                }
            ],
            "version_major" : 1,
            "version_minor" : 0,
            "T" : [[1, 0, 0, 0], [0, -0.259, -0.966, 0.27], [0, 0.966, 0.259, 1.4], [0, 0, 0, 1]]
        }
}
