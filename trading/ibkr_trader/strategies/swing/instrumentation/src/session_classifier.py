from datetime import datetime, time

PRE_START = time(4, 0)
RTH_START = time(9, 30)
RTH_END = time(16, 0)
POST_END = time(20, 0)


class SessionClassifier:
    @staticmethod
    def classify(dt: datetime) -> dict:
        if dt.weekday() >= 5:
            return {"market_session": "WEEKEND", "minutes_into_session": None}

        t = dt.time()
        if RTH_START <= t < RTH_END:
            session = "RTH"
            start = datetime.combine(dt.date(), RTH_START, tzinfo=dt.tzinfo)
        elif PRE_START <= t < RTH_START:
            session = "PRE"
            start = datetime.combine(dt.date(), PRE_START, tzinfo=dt.tzinfo)
        elif RTH_END <= t < POST_END:
            session = "ETH_POST"
            start = datetime.combine(dt.date(), RTH_END, tzinfo=dt.tzinfo)
        else:
            session = "ETH_POST"
            start = datetime.combine(dt.date(), POST_END, tzinfo=dt.tzinfo)

        minutes = int((dt - start).total_seconds() / 60)
        return {"market_session": session, "minutes_into_session": max(0, minutes)}
