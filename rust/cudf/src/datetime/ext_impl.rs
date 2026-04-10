#![allow(clippy::wildcard_imports)]
use super::*;

impl private::Sealed for ColumnView<'_> {}

impl DatetimeExt for ColumnView<'_> {
    fn extract_year(&self) -> ExtractYear<'_> {
        ExtractYear {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_month(&self) -> ExtractMonth<'_> {
        ExtractMonth {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_day(&self) -> ExtractDay<'_> {
        ExtractDay {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_weekday(&self) -> ExtractWeekday<'_> {
        ExtractWeekday {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_hour(&self) -> ExtractHour<'_> {
        ExtractHour {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_minute(&self) -> ExtractMinute<'_> {
        ExtractMinute {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_second(&self) -> ExtractSecond<'_> {
        ExtractSecond {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn day_of_year(&self) -> DayOfYear<'_> {
        DayOfYear {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn is_leap_year(&self) -> IsLeapYear<'_> {
        IsLeapYear {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn days_in_month(&self) -> DaysInMonth<'_> {
        DaysInMonth {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn last_day_of_month(&self) -> LastDayOfMonth<'_> {
        LastDayOfMonth {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_quarter(&self) -> ExtractQuarter<'_> {
        ExtractQuarter {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn dt_ceil(&self, freq: RoundingFrequency) -> DtCeil<'_> {
        DtCeil {
            view: self,
            freq,
            stream: Stream::default_stream(),
        }
    }
    fn dt_floor(&self, freq: RoundingFrequency) -> DtFloor<'_> {
        DtFloor {
            view: self,
            freq,
            stream: Stream::default_stream(),
        }
    }
    fn dt_round(&self, freq: RoundingFrequency) -> DtRound<'_> {
        DtRound {
            view: self,
            freq,
            stream: Stream::default_stream(),
        }
    }
    fn dt_add_months<'a>(&'a self, months: &'a ColumnView<'a>) -> DtAddMonths<'a> {
        DtAddMonths {
            view: self,
            months,
            stream: Stream::default_stream(),
        }
    }
    fn dt_add_months_scalar<'a>(
        &'a self,
        months: &'a crate::scalar::Scalar,
    ) -> DtAddMonthsScalar<'a> {
        DtAddMonthsScalar {
            view: self,
            months,
            stream: Stream::default_stream(),
        }
    }
    fn extract_millisecond(&self) -> ExtractMillisecond<'_> {
        ExtractMillisecond {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_microsecond(&self) -> ExtractMicrosecond<'_> {
        ExtractMicrosecond {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn extract_nanosecond(&self) -> ExtractNanosecond<'_> {
        ExtractNanosecond {
            view: self,
            stream: Stream::default_stream(),
        }
    }
}
