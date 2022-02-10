def fix_team_names(row):
    if row in ["New Jersey Nets", "New York Nets"]:
        return "Brooklyn Nets"
    if row in ["Washington Bullets", "Baltimore Bullets", "Capital Bullets"]:
        return "Washington Wizards"
    if row in ["LA Clippers", "San Diego Clippers", "Buffalo Braves"]:
        return "Los Angeles Clippers"
    if row in ["Kansas City Kings", "Cincinnati Royals", "Kansas City"]:
        return "Sacramento Kings"
    if row == "Seattle SuperSonics":
        return "Oklahoma City Thunder"
    if row in ["New Orleans/Oklahoma City Hornets", "New Orleans Hornets"]:
        return "New Orleans Pelicans"
    if row == "Charlotte Bobcats":
        return "Charlotte Hornets"
    if row == "Vancouver Grizzlies":
        return "Memphis Grizzlies"
    if row == "San Francisco Warriors":
        return "Golden State Warriors"
    if row == "San Diego Rockets":
        return "Houston Rockets"
    if row == "New Orleans Jazz":
        return "Utah Jazz"
    return row


def get_names(row):
    for i, v in enumerate(row):
        if v == '-':
            return row[:i].strip()
    return row
