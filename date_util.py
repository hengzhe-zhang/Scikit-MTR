import calendar


def replace_abbreviated_months_with_numbers(df, column_name):
    if "jan" not in df[column_name]:
        return
    month_to_number = {
        month.lower(): index for index, month in enumerate(calendar.month_abbr) if month
    }
    df[column_name] = df[column_name].str.lower().map(month_to_number)


def replace_abbreviated_days_with_numbers(df, column_name):
    if "mon" not in df[column_name]:
        return
    day_to_number = {day.lower(): index for index, day in enumerate(calendar.day_abbr)}
    df[column_name] = df[column_name].str.lower().map(day_to_number)
