// Copyright © 2026 Apple Inc.

import AppKit
import CoreImage
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Testing
import Tokenizers

extension ModelLoadingSuite {
    @Suite(.serialized) struct Idefics3IntegrationTests {}
}

extension ModelLoadingSuite.Idefics3IntegrationTests {
    /// granite-docling declares `Idefics3Processor` with 512 px tiles and its own image token.
    @Test func graniteDoclingReadsTextFromAPage() async throws {
        let response = try await respond(
            modelID: "ibm-granite/granite-docling-258M-mlx",
            prompt: "Convert this page to docling.",
            image: page(text: "Quarterly Report"), maxTokens: 64)
        #expect(response.contains("Quarterly Report"), "got: \(response)")
    }

    /// SmolVLM (v1) also declares `Idefics3Processor`, with 81 image tokens per tile.
    @Test func smolVLMNamesTheImageColor() async throws {
        let response = try await respond(
            modelID: "mlx-community/SmolVLM-Instruct-4bit",
            prompt: "What color is this image? Reply with just the color name.",
            image: VisionTestImages.solidColor(.red), maxTokens: 8)
        #expect(response.lowercased().contains("red"), "got: \(response)")
    }

    private func respond(
        modelID: String, prompt: String, image: CIImage, maxTokens: Int
    ) async throws -> String {
        let container = try await VLMModelFactory.shared.loadContainer(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: modelID))
        let session = ChatSession(
            container, generateParameters: GenerateParameters(maxTokens: maxTokens, temperature: 0))
        return try await session.respond(to: prompt, image: .ciImage(image))
    }

    private func page(text: String) -> CIImage {
        let size = NSSize(width: 612, height: 792)
        let image = NSImage(size: size, flipped: false) { rect in
            NSColor.white.setFill()
            rect.fill()
            (text as NSString).draw(
                at: NSPoint(x: 72, y: 680),
                withAttributes: [.font: NSFont.boldSystemFont(ofSize: 36)])
            return true
        }
        let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil)!
        return CIImage(cgImage: cgImage)
    }
}
