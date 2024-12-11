import calendar


def replace_abbreviated_months_with_numbers(df, column_name):
    name_list = list(df[column_name].unique())
    if not ("jan" in name_list or "feb" in name_list):
        return
    month_to_number = {
        month.lower(): index for index, month in enumerate(calendar.month_abbr) if month
    }
    df[column_name] = df[column_name].str.lower().map(month_to_number)


def replace_abbreviated_days_with_numbers(df, column_name):
    name_list = list(df[column_name].unique())
    if not ("mon" in name_list or "tue" in name_list):
        return
    day_to_number = {day.lower(): index for index, day in enumerate(calendar.day_abbr)}
    df[column_name] = df[column_name].str.lower().map(day_to_number)
