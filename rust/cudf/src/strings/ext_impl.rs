#![allow(clippy::wildcard_imports)]
use super::*;

impl StringExt for ColumnView<'_> {
    fn to_lower(&self) -> ToLower<'_> {
        ToLower {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn to_upper(&self) -> ToUpper<'_> {
        ToUpper {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_contains<'a>(&'a self, target: &'a Scalar) -> StrContains<'a> {
        StrContains {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_starts_with<'a>(&'a self, target: &'a Scalar) -> StrStartsWith<'a> {
        StrStartsWith {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_ends_with<'a>(&'a self, target: &'a Scalar) -> StrEndsWith<'a> {
        StrEndsWith {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_find<'a>(&'a self, target: &'a Scalar) -> StrFind<'a> {
        StrFind {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_replace<'a>(&'a self, target: &'a Scalar, replacement: &'a Scalar) -> StrReplace<'a> {
        StrReplace {
            view: self,
            target,
            replacement,
            stream: Stream::default_stream(),
        }
    }
    fn str_strip(&self) -> StrStrip<'_> {
        StrStrip {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_lstrip(&self) -> StrLstrip<'_> {
        StrLstrip {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_rstrip(&self) -> StrRstrip<'_> {
        StrRstrip {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn count_characters(&self) -> CountCharacters<'_> {
        CountCharacters {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn count_bytes(&self) -> CountBytes<'_> {
        CountBytes {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_from_integers(&self) -> StrFromIntegers<'_> {
        StrFromIntegers {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_to_integers(&self, output_type: TypeId) -> StrToIntegers<'_> {
        StrToIntegers {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }
    fn str_from_floats(&self) -> StrFromFloats<'_> {
        StrFromFloats {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_to_floats(&self, output_type: TypeId) -> StrToFloats<'_> {
        StrToFloats {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }
    fn str_pad<'a>(&'a self, width: usize, side: SideType, fill_char: &'a str) -> StrPad<'a> {
        StrPad {
            view: self,
            width,
            side,
            fill_char,
            stream: Stream::default_stream(),
        }
    }
    fn str_zfill(&self, width: usize) -> StrZfill<'_> {
        StrZfill {
            view: self,
            width,
            stream: Stream::default_stream(),
        }
    }
    fn str_zfill_by_widths<'a>(&'a self, widths: &'a ColumnView<'a>) -> StrZfillByWidths<'a> {
        StrZfillByWidths {
            view: self,
            widths,
            stream: Stream::default_stream(),
        }
    }
    fn str_slice(&self, start: i32, stop: i32, step: i32) -> StrSlice<'_> {
        StrSlice {
            view: self,
            start,
            stop,
            step,
            stream: Stream::default_stream(),
        }
    }
    fn str_repeat(&self, times: usize) -> StrRepeat<'_> {
        StrRepeat {
            view: self,
            times,
            stream: Stream::default_stream(),
        }
    }
    fn str_split<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32) -> StrSplit<'a> {
        StrSplit {
            view: self,
            delimiter,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_rsplit<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32) -> StrRsplit<'a> {
        StrRsplit {
            view: self,
            delimiter,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_split_part<'a>(&'a self, delimiter: &'a Scalar, index: i32) -> StrSplitPart<'a> {
        StrSplitPart {
            view: self,
            delimiter,
            index,
            stream: Stream::default_stream(),
        }
    }
    fn str_join<'a>(&'a self, separator: &'a Scalar, narep: &'a Scalar) -> StrJoin<'a> {
        StrJoin {
            view: self,
            separator,
            narep,
            stream: Stream::default_stream(),
        }
    }
    fn str_like<'a>(&'a self, pattern: &'a str, escape_char: &'a str) -> StrLike<'a> {
        StrLike {
            view: self,
            pattern,
            escape_char,
            stream: Stream::default_stream(),
        }
    }
    fn str_contains_re<'a>(&'a self, pattern: &'a str) -> StrContainsRe<'a> {
        StrContainsRe {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn str_matches_re<'a>(&'a self, pattern: &'a str) -> StrMatchesRe<'a> {
        StrMatchesRe {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn str_count_re<'a>(&'a self, pattern: &'a str) -> StrCountRe<'a> {
        StrCountRe {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn str_replace_re<'a>(&'a self, pattern: &'a str, replacement: &'a str) -> StrReplaceRe<'a> {
        StrReplaceRe {
            view: self,
            pattern,
            replacement,
            stream: Stream::default_stream(),
        }
    }
    fn swapcase(&self) -> Swapcase<'_> {
        Swapcase {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_strip_chars<'a>(&'a self, side: SideType, to_strip: &'a str) -> StrStripChars<'a> {
        StrStripChars {
            view: self,
            side,
            to_strip,
            stream: Stream::default_stream(),
        }
    }
    fn str_replace_literal<'a>(
        &'a self,
        target: &'a str,
        repl: &'a str,
        maxrepl: i32,
    ) -> StrReplaceLiteral<'a> {
        StrReplaceLiteral {
            view: self,
            target,
            repl,
            maxrepl,
            stream: Stream::default_stream(),
        }
    }
    fn str_find_str<'a>(&'a self, target: &'a str, start: i32, stop: i32) -> StrFindStr<'a> {
        StrFindStr {
            view: self,
            target,
            start,
            stop,
            stream: Stream::default_stream(),
        }
    }
    fn str_rfind<'a>(&'a self, target: &'a str, start: i32, stop: i32) -> StrRfind<'a> {
        StrRfind {
            view: self,
            target,
            start,
            stop,
            stream: Stream::default_stream(),
        }
    }
    fn str_contains_literal<'a>(&'a self, target: &'a str) -> StrContainsLiteral<'a> {
        StrContainsLiteral {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_starts_with_str<'a>(&'a self, target: &'a str) -> StrStartsWithStr<'a> {
        StrStartsWithStr {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_ends_with_str<'a>(&'a self, target: &'a str) -> StrEndsWithStr<'a> {
        StrEndsWithStr {
            view: self,
            target,
            stream: Stream::default_stream(),
        }
    }
    fn str_reverse(&self) -> StrReverse<'_> {
        StrReverse {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_extract<'a>(&'a self, pattern: &'a str) -> StrExtract<'a> {
        StrExtract {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn str_extract_all<'a>(&'a self, pattern: &'a str) -> StrExtractAll<'a> {
        StrExtractAll {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn str_findall<'a>(&'a self, pattern: &'a str) -> StrFindall<'a> {
        StrFindall {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn str_find_re<'a>(&'a self, pattern: &'a str) -> StrFindRe<'a> {
        StrFindRe {
            view: self,
            pattern,
            stream: Stream::default_stream(),
        }
    }
    fn capitalize(&self) -> Capitalize<'_> {
        Capitalize {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn title(&self) -> Title<'_> {
        Title {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn is_title(&self) -> IsTitle<'_> {
        IsTitle {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn wrap(&self, width: i32) -> Wrap<'_> {
        Wrap {
            view: self,
            width,
            stream: Stream::default_stream(),
        }
    }
    fn str_to_timestamps<'a>(
        &'a self,
        timestamp_type: TypeId,
        format: &'a str,
    ) -> StrToTimestamps<'a> {
        StrToTimestamps {
            view: self,
            timestamp_type,
            format,
            stream: Stream::default_stream(),
        }
    }
    fn str_from_timestamps<'a>(&'a self, format: &'a str) -> StrFromTimestamps<'a> {
        StrFromTimestamps {
            view: self,
            format,
            stream: Stream::default_stream(),
        }
    }
    fn str_is_timestamp<'a>(&'a self, format: &'a str) -> StrIsTimestamp<'a> {
        StrIsTimestamp {
            view: self,
            format,
            stream: Stream::default_stream(),
        }
    }
    fn str_to_booleans<'a>(&'a self, true_string: &'a str) -> StrToBooleans<'a> {
        StrToBooleans {
            view: self,
            true_string,
            stream: Stream::default_stream(),
        }
    }
    fn str_from_booleans<'a>(
        &'a self,
        true_string: &'a str,
        false_string: &'a str,
    ) -> StrFromBooleans<'a> {
        StrFromBooleans {
            view: self,
            true_string,
            false_string,
            stream: Stream::default_stream(),
        }
    }
    fn str_to_durations<'a>(
        &'a self,
        duration_type: TypeId,
        format: &'a str,
    ) -> StrToDurations<'a> {
        StrToDurations {
            view: self,
            duration_type,
            format,
            stream: Stream::default_stream(),
        }
    }
    fn str_from_durations<'a>(&'a self, format: &'a str) -> StrFromDurations<'a> {
        StrFromDurations {
            view: self,
            format,
            stream: Stream::default_stream(),
        }
    }
    fn str_to_fixed_point(&self, type_id: TypeId, scale: i32) -> StrToFixedPoint<'_> {
        StrToFixedPoint {
            view: self,
            type_id,
            scale,
            stream: Stream::default_stream(),
        }
    }
    fn str_from_fixed_point(&self) -> StrFromFixedPoint<'_> {
        StrFromFixedPoint {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_is_fixed_point(&self, type_id: TypeId, scale: i32) -> StrIsFixedPoint<'_> {
        StrIsFixedPoint {
            view: self,
            type_id,
            scale,
            stream: Stream::default_stream(),
        }
    }
    fn url_encode(&self) -> UrlEncode<'_> {
        UrlEncode {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn url_decode(&self) -> UrlDecode<'_> {
        UrlDecode {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn ipv4_to_integers(&self) -> Ipv4ToIntegers<'_> {
        Ipv4ToIntegers {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn integers_to_ipv4(&self) -> IntegersToIpv4<'_> {
        IntegersToIpv4 {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn is_ipv4(&self) -> IsIpv4<'_> {
        IsIpv4 {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_split_re<'a>(&'a self, pattern: &'a str, maxsplit: i32) -> StrSplitRe<'a> {
        StrSplitRe {
            view: self,
            pattern,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_rsplit_re<'a>(&'a self, pattern: &'a str, maxsplit: i32) -> StrRsplitRe<'a> {
        StrRsplitRe {
            view: self,
            pattern,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_split_record_re<'a>(&'a self, pattern: &'a str, maxsplit: i32) -> StrSplitRecordRe<'a> {
        StrSplitRecordRe {
            view: self,
            pattern,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_rsplit_record_re<'a>(
        &'a self,
        pattern: &'a str,
        maxsplit: i32,
    ) -> StrRsplitRecordRe<'a> {
        StrRsplitRecordRe {
            view: self,
            pattern,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_partition<'a>(&'a self, delimiter: &'a str) -> StrPartition<'a> {
        StrPartition {
            view: self,
            delimiter,
            stream: Stream::default_stream(),
        }
    }
    fn str_rpartition<'a>(&'a self, delimiter: &'a str) -> StrRpartition<'a> {
        StrRpartition {
            view: self,
            delimiter,
            stream: Stream::default_stream(),
        }
    }
    fn str_replace_with_backrefs<'a>(
        &'a self,
        pattern: &'a str,
        replacement: &'a str,
    ) -> StrReplaceWithBackrefs<'a> {
        StrReplaceWithBackrefs {
            view: self,
            pattern,
            replacement,
            stream: Stream::default_stream(),
        }
    }
    fn str_repeat_column<'a>(&'a self, repeat_times: &'a ColumnView<'a>) -> StrRepeatColumn<'a> {
        StrRepeatColumn {
            view: self,
            repeat_times,
            stream: Stream::default_stream(),
        }
    }
    fn str_contains_multiple<'a>(&'a self, targets: &'a ColumnView<'a>) -> StrContainsMultiple<'a> {
        StrContainsMultiple {
            view: self,
            targets,
            stream: Stream::default_stream(),
        }
    }
    fn str_find_multiple<'a>(&'a self, targets: &'a ColumnView<'a>) -> StrFindMultiple<'a> {
        StrFindMultiple {
            view: self,
            targets,
            stream: Stream::default_stream(),
        }
    }
    fn all_characters_of_type(&self, types: u32, verify_types: u32) -> AllCharactersOfType<'_> {
        AllCharactersOfType {
            view: self,
            types,
            verify_types,
            stream: Stream::default_stream(),
        }
    }
    fn filter_characters_of_type<'a>(
        &'a self,
        types_to_remove: u32,
        replacement: &'a str,
        types_to_keep: u32,
    ) -> FilterCharactersOfType<'a> {
        FilterCharactersOfType {
            view: self,
            types_to_remove,
            replacement,
            types_to_keep,
            stream: Stream::default_stream(),
        }
    }
    fn str_is_integer(&self) -> StrIsInteger<'_> {
        StrIsInteger {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_is_integer_with_type(&self, int_type: TypeId) -> StrIsIntegerWithType<'_> {
        StrIsIntegerWithType {
            view: self,
            int_type,
            stream: Stream::default_stream(),
        }
    }
    fn str_is_float(&self) -> StrIsFloat<'_> {
        StrIsFloat {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_hex_to_integers(&self, output_type: TypeId) -> StrHexToIntegers<'_> {
        StrHexToIntegers {
            view: self,
            output_type,
            stream: Stream::default_stream(),
        }
    }
    fn str_is_hex(&self) -> StrIsHex<'_> {
        StrIsHex {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_integers_to_hex(&self) -> StrIntegersToHex<'_> {
        StrIntegersToHex {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_replace_slice<'a>(
        &'a self,
        repl: &'a str,
        start: i32,
        stop: i32,
    ) -> StrReplaceSlice<'a> {
        StrReplaceSlice {
            view: self,
            repl,
            start,
            stop,
            stream: Stream::default_stream(),
        }
    }
    fn str_replace_multiple<'a>(
        &'a self,
        targets: &'a ColumnView<'a>,
        repls: &'a ColumnView<'a>,
    ) -> StrReplaceMultiple<'a> {
        StrReplaceMultiple {
            view: self,
            targets,
            repls,
            stream: Stream::default_stream(),
        }
    }
    fn str_split_record<'a>(&'a self, delimiter: &'a Scalar, maxsplit: i32) -> StrSplitRecord<'a> {
        StrSplitRecord {
            view: self,
            delimiter,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_rsplit_record<'a>(
        &'a self,
        delimiter: &'a Scalar,
        maxsplit: i32,
    ) -> StrRsplitRecord<'a> {
        StrRsplitRecord {
            view: self,
            delimiter,
            maxsplit,
            stream: Stream::default_stream(),
        }
    }
    fn str_join_list_elements<'a>(
        &'a self,
        separator: &'a str,
        narep: &'a str,
    ) -> StrJoinListElements<'a> {
        StrJoinListElements {
            view: self,
            separator,
            narep,
            stream: Stream::default_stream(),
        }
    }
    fn str_translate<'a>(&'a self, from_chars: &'a [u32], to_chars: &'a [u32]) -> StrTranslate<'a> {
        StrTranslate {
            view: self,
            from_chars,
            to_chars,
            stream: Stream::default_stream(),
        }
    }
    fn str_filter_characters<'a>(
        &'a self,
        from_chars: &'a [u32],
        to_chars: &'a [u32],
        keep: bool,
        replacement: &'a str,
    ) -> StrFilterCharacters<'a> {
        StrFilterCharacters {
            view: self,
            from_chars,
            to_chars,
            keep,
            replacement,
            stream: Stream::default_stream(),
        }
    }
    fn code_points(&self) -> CodePoints<'_> {
        CodePoints {
            view: self,
            stream: Stream::default_stream(),
        }
    }
    fn str_cast_to_integer(&self, output_type: TypeId, big_endian: bool) -> StrCastToInteger<'_> {
        StrCastToInteger {
            view: self,
            output_type,
            big_endian,
            stream: Stream::default_stream(),
        }
    }
    fn str_cast_from_integer(&self, big_endian: bool) -> StrCastFromInteger<'_> {
        StrCastFromInteger {
            view: self,
            big_endian,
            stream: Stream::default_stream(),
        }
    }
    fn str_slice_column<'a>(
        &'a self,
        starts: &'a ColumnView<'a>,
        stops: &'a ColumnView<'a>,
    ) -> StrSliceColumn<'a> {
        StrSliceColumn {
            view: self,
            starts,
            stops,
            stream: Stream::default_stream(),
        }
    }
    fn str_extract_single<'a>(
        &'a self,
        pattern: &'a str,
        group_index: i32,
    ) -> StrExtractSingle<'a> {
        StrExtractSingle {
            view: self,
            pattern,
            group_index,
            stream: Stream::default_stream(),
        }
    }
    fn str_join_list_elements_column<'a>(
        &'a self,
        separators: &'a ColumnView<'a>,
        separator_narep: &'a str,
        string_narep: &'a str,
    ) -> StrJoinListElementsColumn<'a> {
        StrJoinListElementsColumn {
            view: self,
            separators,
            separator_narep,
            string_narep,
            stream: Stream::default_stream(),
        }
    }
    fn str_join_strings<'a>(&'a self, separator: &'a str, narep: &'a str) -> StrJoinStrings<'a> {
        StrJoinStrings {
            view: self,
            separator,
            narep,
            stream: Stream::default_stream(),
        }
    }
    fn str_find_instance<'a>(&'a self, target: &'a str, instance: i32) -> StrFindInstance<'a> {
        StrFindInstance {
            view: self,
            target,
            instance,
            stream: Stream::default_stream(),
        }
    }
    fn str_like_column<'a>(
        &'a self,
        patterns: &'a ColumnView<'a>,
        escape_char: &'a str,
    ) -> StrLikeColumn<'a> {
        StrLikeColumn {
            view: self,
            patterns,
            escape_char,
            stream: Stream::default_stream(),
        }
    }
}
