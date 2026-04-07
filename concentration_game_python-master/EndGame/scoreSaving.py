

def scoreWrite(first_name, last_name, score, level=1):
    """Saves the scores into files, 1 file by level

    :param first_name: first name input
    :param last_name:  last name input
    :param score: score storage
    :param level: level chosen
    :return:
    """
    if level == 1:
        f = open("score_level_1.txt", "a")
    if level == 2:
        f = open("score_level_2.txt", "a")
    if level == 3:
        f = open("score_level_3.txt", "a")
    total_score = 0
    for i in score: #loop to add the final score
        total_score += score[i]

    f.write(first_name + " " + last_name + f" Score = {total_score:.2f} on level {level}\n")
    f.close()
