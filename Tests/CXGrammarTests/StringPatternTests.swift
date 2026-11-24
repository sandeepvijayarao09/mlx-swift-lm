// Copyright © 2026 Apple Inc.

import Foundation
import MLXCXGrammar
import Testing

@Suite
struct StringPatternTests {

    /// A top-level alternation in `pattern` must still produce a quoted string.
    /// https://github.com/ml-explore/mlx-swift-lm/issues/828
    @Test(arguments: [
        (#"{"total":"49.35"}"#, true),
        (#"{"total":"-5"}"#, true),
        (#"{"total":"-"}"#, true),
        (#"{"total":49.35"}"#, false),
        (#"{"total":"-}"#, false),
        (#"{"total":"abc"}"#, false),
    ])
    func testAlternationPatternIsQuoted(text: String, accepted: Bool) throws {
        let schema = #"""
            {"type":"object","properties":{"total":{"type":"string","pattern":"-|-?[0-9]+(\\.[0-9]+)?"}},"required":["total"]}
            """#
        #expect(try Self.grammar(for: schema, accepts: text) == accepted)
    }

    // MARK: - Helpers

    /// One token per printable ASCII character, plus a stop token.
    private static let characters: [String] = (0x20 ... 0x7E).map {
        String(UnicodeScalar(UInt8($0)))
    }
    private static let stopToken = "</s>"

    /// Whether the grammar for `schema` accepts exactly `text`, then the stop token.
    private static func grammar(for schema: String, accepts text: String) throws -> Bool {
        let vocab = characters + [stopToken]
        let stopId = Int32(vocab.count - 1)

        let cStrings = vocab.map { $0.utf8CString }
        let vocabPtrs: [UnsafePointer<CChar>?] = cStrings.map { arr in
            arr.withUnsafeBufferPointer { $0.baseAddress }
        }

        var info: OpaquePointer?
        let stopTokens = [stopId]
        let infoStatus: XGStatus = vocabPtrs.withUnsafeBufferPointer { vocabBuf in
            stopTokens.withUnsafeBufferPointer { stopBuf in
                xg_tokenizer_info_new(
                    vocabBuf.baseAddress, vocabBuf.count, XG_VOCAB_TYPE_RAW,
                    stopBuf.baseAddress, stopBuf.count, &info)
            }
        }
        try #require(infoStatus == XG_OK)
        defer { xg_tokenizer_info_free(info) }

        var compiler: OpaquePointer?
        try #require(xg_grammar_compiler_new(info, &compiler) == XG_OK)
        defer { xg_grammar_compiler_free(compiler) }

        var compiled: OpaquePointer?
        let compileStatus = schema.withCString { xg_compile_json_schema(compiler, $0, &compiled) }
        try #require(
            compileStatus == XG_OK,
            "xg_compile_json_schema failed: \(xg_last_error_message().map { String(cString: $0) } ?? "<nil>")"
        )
        defer { xg_compiled_grammar_free(compiled) }

        var matcher: OpaquePointer?
        try #require(xg_matcher_new(compiled, &matcher) == XG_OK)
        defer { xg_matcher_free(matcher) }

        for character in text {
            let id = try #require(characters.firstIndex(of: String(character)))
            if xg_matcher_accept_token(matcher, Int32(id)) != XG_OK {
                return false
            }
        }
        if xg_matcher_accept_token(matcher, stopId) != XG_OK {
            return false
        }
        var terminated: Int32 = 0
        try #require(xg_matcher_is_terminated(matcher, &terminated) == XG_OK)
        return terminated == 1
    }
}
