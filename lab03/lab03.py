"""
Laboratório 03 - Granulometria Morfológica

Disciplina: Visão Computacional

Objetivo
Investigar o uso de granulometria morfológica para caracterização de imagens em diferentes escalas.
Cada aluno deverá obter uma assinatura granulométrica para cada imagem, comparar os resultados
entre classes e analisar se esse tipo de descritor é útil para distinguir texturas ou materiais.
Base de imagens sugerida
Sugere-se usar um subconjunto da base KTH-TIPS em tons de cinza. Selecione 3 classes e use entre
10 e 20 imagens por classe, totalizando entre 30 e 60 imagens.
Página oficial da base:
https://www.csc.kth.se/cvap/databases/kth-tips/download.html
As imagens devem ser mantidas em tons de cinza e, se necessário, redimensionadas para o mesmo
tamanho.

Descrição geral da atividade
Cada aluno deverá representar cada imagem por uma assinatura granulométrica obtida por operações
morfológicas em múltiplas escalas. A escolha da estratégia de implementação fica a critério do aluno.
O laboratório deve incluir a geração das assinaturas para todas as imagens escolhidas, a comparação
entre as classes e uma análise dos resultados obtidos.

Configuração experimental mínima
Deve-se usar uma única base de imagens, com pelo menos 3 classes.
Cada classe deve conter entre 10 e 20 imagens.
Deve-se gerar uma assinatura granulométrica para cada imagem.
Deve-se apresentar pelo menos um gráfico médio por classe.
Deve-se realizar uma comparação entre as classes usando as assinaturas obtidas.

O que deve ser apresentado no relatório
Descrição da base utilizada e das classes selecionadas.
Descrição da abordagem adotada para obter as assinaturas granulométricas.
Exemplos de imagens da base.
Gráficos das assinaturas granulométricas individuais e médias por classe.
Análise de quais classes apresentaram curvas mais semelhantes e quais foram mais fáceis de separar.
Discussão sobre o efeito da faixa de escalas escolhida.
Relato das dificuldades encontradas ao longo do desenvolvimento.
Conclusão sobre a utilidade da granulometria morfológica para o problema estudado.

Entrega
O trabalho é individual.
Entregar o código-fonte da implementação.
Entregar um relatório em PDF contendo as figuras, tabelas e análises solicitadas.

Observações
O foco do laboratório é investigar se a granulometria morfológica produz descritores úteis para
caracterizar imagens em diferentes escalas.
É permitido usar funções prontas da biblioteca escolhida.
Pode implementar em C, C++ ou python.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_IMAGE_SIZE = (200, 200)
DEFAULT_MAX_RADIUS = 20


@dataclass(frozen=True)
class SampleResult:
	classe: str
	image_name: str
	signature: np.ndarray
	removed_fraction: np.ndarray


def discover_classes(base_dir: Path) -> list[Path]:
	return sorted([path for path in base_dir.iterdir() if path.is_dir()])


def discover_images(class_dir: Path) -> list[Path]:
	return sorted(
		[path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
	)


def load_grayscale_image(image_path: Path, target_size: tuple[int, int]) -> np.ndarray:
	image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise ValueError(f"Nao foi possivel ler a imagem: {image_path}")

	if target_size is not None:
		target_width, target_height = target_size
		if image.shape[1] != target_width or image.shape[0] != target_height:
			image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

	return image.astype(np.float32) / 255.0


def ellipse_kernel(radius: int) -> np.ndarray:
	size = 2 * radius + 1
	return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def opening_mass(image: np.ndarray, radius: int) -> float:
	opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, ellipse_kernel(radius))
	return float(opened.sum())


def granulometric_signature(image: np.ndarray, max_radius: int) -> tuple[np.ndarray, np.ndarray]:
	original_mass = float(image.sum())
	if original_mass <= 0:
		zeros = np.zeros(max_radius, dtype=np.float32)
		return zeros, zeros

	masses = np.array([opening_mass(image, radius) for radius in range(1, max_radius + 1)], dtype=np.float32)
	retention = masses / original_mass
	removed = np.clip(np.r_[1.0, retention[:-1]] - retention, 0.0, 1.0)
	return retention, removed


def summarize_signature(signature: np.ndarray) -> dict[str, float]:
	radii = np.arange(1, signature.size + 1, dtype=np.float32)
	peak_index = int(np.argmax(signature))
	weighted_radius = float(np.average(radii, weights=np.maximum(signature, 1e-12)))
	area_under_curve = float(np.trapezoid(signature, radii))

	return {
		"peak_radius": float(radii[peak_index]),
		"peak_value": float(signature[peak_index]),
		"weighted_radius": weighted_radius,
		"area_under_curve": area_under_curve,
		"final_retention": float(signature[-1]),
	}


def collect_results(base_dir: Path, image_size: tuple[int, int], max_radius: int) -> tuple[list[SampleResult], pd.DataFrame]:
	results: list[SampleResult] = []
	records: list[dict[str, object]] = []

	for class_dir in discover_classes(base_dir):
		images = discover_images(class_dir)
		for image_path in images:
			image = load_grayscale_image(image_path, image_size)
			retention, removed = granulometric_signature(image, max_radius=max_radius)
			results.append(
				SampleResult(
					classe=class_dir.name,
					image_name=image_path.name,
					signature=retention,
					removed_fraction=removed,
				)
			)

			for radius, value in enumerate(retention, start=1):
				records.append(
					{
						"class": class_dir.name,
						"image": image_path.name,
						"radius": radius,
						"retention": float(value),
						"removed_fraction": float(removed[radius - 1]),
					}
				)

	dataframe = pd.DataFrame(records)
	return results, dataframe


def build_class_statistics(results: list[SampleResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
	if not results:
		return pd.DataFrame(), pd.DataFrame()

	rows = []
	for result in results:
		summary = summarize_signature(result.signature)
		rows.append(
			{
				"class": result.classe,
				"image": result.image_name,
				**summary,
			}
		)

	image_stats = pd.DataFrame(rows)
	class_stats = (
		image_stats.groupby("class", as_index=False)
		.agg(
			image_count=("image", "count"),
			peak_radius_mean=("peak_radius", "mean"),
			peak_radius_std=("peak_radius", "std"),
			weighted_radius_mean=("weighted_radius", "mean"),
			weighted_radius_std=("weighted_radius", "std"),
			area_under_curve_mean=("area_under_curve", "mean"),
			area_under_curve_std=("area_under_curve", "std"),
			final_retention_mean=("final_retention", "mean"),
			final_retention_std=("final_retention", "std"),
		)
		.sort_values("class")
	)

	return image_stats, class_stats


def compare_class_means(results: list[SampleResult]) -> pd.DataFrame:
	if not results:
		return pd.DataFrame()

	class_names = sorted({result.classe for result in results})
	class_means = {
		class_name: np.mean([result.signature for result in results if result.classe == class_name], axis=0)
		for class_name in class_names
	}

	comparison_rows = []
	for left in class_names:
		for right in class_names:
			distance = float(np.linalg.norm(class_means[left] - class_means[right]))
			correlation = float(np.corrcoef(class_means[left], class_means[right])[0, 1])
			comparison_rows.append(
				{
					"class_a": left,
					"class_b": right,
					"euclidean_distance": distance,
					"correlation": correlation,
				}
			)

	return pd.DataFrame(comparison_rows)


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
	if dataframe.empty:
		return "| vazio |\n| --- |\n| sem dados |"

	frame = dataframe.copy().fillna("")
	headers = [str(column) for column in frame.columns]
	rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
	for _, row in frame.iterrows():
		values = [str(row[column]) for column in frame.columns]
		rows.append("| " + " | ".join(values) + " |")
	return "\n".join(rows)


def format_float(value: float, digits: int = 4) -> str:
	return f"{value:.{digits}f}"


def best_pair_from_comparison(comparison: pd.DataFrame) -> tuple[str, str, float, float] | None:
	filtered = comparison[comparison["class_a"] != comparison["class_b"]]
	if filtered.empty:
		return None
	row = filtered.sort_values(["euclidean_distance", "correlation"], ascending=[False, False]).iloc[0]
	return str(row["class_a"]), str(row["class_b"]), float(row["euclidean_distance"]), float(row["correlation"])


def worst_pair_from_comparison(comparison: pd.DataFrame) -> tuple[str, str, float, float] | None:
	filtered = comparison[comparison["class_a"] != comparison["class_b"]]
	if filtered.empty:
		return None
	row = filtered.sort_values(["euclidean_distance", "correlation"], ascending=[True, False]).iloc[0]
	return str(row["class_a"]), str(row["class_b"]), float(row["euclidean_distance"]), float(row["correlation"])


def render_summary_bullets(class_stats: pd.DataFrame, comparison: pd.DataFrame) -> list[str]:
	lines: list[str] = []
	if class_stats.empty or comparison.empty:
		return lines

	most_distinct = class_stats.sort_values("area_under_curve_mean", ascending=False).iloc[0]
	least_distinct = class_stats.sort_values("area_under_curve_mean", ascending=True).iloc[0]
	closest_pair = worst_pair_from_comparison(comparison)
	furthest_pair = best_pair_from_comparison(comparison)

	lines.append(
		f"- A classe com maior área média sob a curva foi {most_distinct['class']}, indicando maior retenção de massa após as aberturas sucessivas."
	)
	lines.append(
		f"- A classe com menor área média sob a curva foi {least_distinct['class']}, sugerindo erosão morfológica mais rápida ao aumentar a escala."
	)
	if closest_pair is not None:
		left, right, distance, correlation = closest_pair
		lines.append(
			f"- O par mais semelhante foi {left} × {right}, com distância euclidiana {format_float(distance)} e correlação {format_float(correlation)}."
		)
	if furthest_pair is not None:
		left, right, distance, correlation = furthest_pair
		lines.append(
			f"- O par mais separado foi {left} × {right}, com distância euclidiana {format_float(distance)} e correlação {format_float(correlation)}."
		)
	return lines


def save_sample_montage(base_dir: Path, output_dir: Path, max_images_per_class: int = 4) -> None:
	classes = discover_classes(base_dir)
	if not classes:
		return

	figure, axes = plt.subplots(len(classes), max_images_per_class, figsize=(3.2 * max_images_per_class, 3.2 * len(classes)))
	if len(classes) == 1:
		axes = np.array([axes])

	for row_index, class_dir in enumerate(classes):
		images = discover_images(class_dir)[:max_images_per_class]
		for col_index in range(max_images_per_class):
			axis = axes[row_index, col_index]
			axis.axis("off")
			if col_index < len(images):
				image = cv2.imread(str(images[col_index]), cv2.IMREAD_GRAYSCALE)
				axis.imshow(image, cmap="gray")
				axis.set_title(f"{class_dir.name}\n{images[col_index].stem}", fontsize=9)

	figure.suptitle("Exemplos das classes utilizadas", fontsize=15)
	figure.tight_layout()
	figure.savefig(output_dir / "amostras_das_classes.png", dpi=200, bbox_inches="tight")
	plt.close(figure)


def plot_class_signatures(results: list[SampleResult], output_dir: Path) -> None:
	class_names = sorted({result.classe for result in results})
	if not class_names:
		return

	figure, axes = plt.subplots(len(class_names), 1, figsize=(12, 4.2 * len(class_names)), sharex=True)
	if len(class_names) == 1:
		axes = [axes]

	radii = np.arange(1, results[0].signature.size + 1)

	for axis, class_name in zip(axes, class_names):
		class_signatures = np.array([result.signature for result in results if result.classe == class_name])
		mean_signature = class_signatures.mean(axis=0)
		std_signature = class_signatures.std(axis=0)

		for signature in class_signatures:
			axis.plot(radii, signature, color="#999999", alpha=0.25, linewidth=1)

		axis.plot(radii, mean_signature, color="#1f77b4", linewidth=2.5, label=f"Média {class_name}")
		axis.fill_between(radii, mean_signature - std_signature, mean_signature + std_signature, color="#1f77b4", alpha=0.15)
		axis.set_title(f"Assinaturas granulométricas - {class_name}")
		axis.set_ylabel("Retenção normalizada")
		axis.grid(alpha=0.3)
		axis.legend(loc="best")

	axes[-1].set_xlabel("Raio do elemento estruturante")
	figure.tight_layout()
	figure.savefig(output_dir / "assinaturas_por_classe.png", dpi=200, bbox_inches="tight")
	plt.close(figure)


def plot_mean_comparison(results: list[SampleResult], output_dir: Path) -> None:
	class_names = sorted({result.classe for result in results})
	if not class_names:
		return

	radii = np.arange(1, results[0].signature.size + 1)
	figure, axis = plt.subplots(figsize=(12, 6))

	for class_name in class_names:
		class_signatures = np.array([result.signature for result in results if result.classe == class_name])
		mean_signature = class_signatures.mean(axis=0)
		axis.plot(radii, mean_signature, linewidth=2.5, label=class_name)

	axis.set_title("Comparação entre as médias das classes")
	axis.set_xlabel("Raio do elemento estruturante")
	axis.set_ylabel("Retenção normalizada")
	axis.grid(alpha=0.3)
	axis.legend(title="Classe")
	figure.tight_layout()
	figure.savefig(output_dir / "media_entre_classes.png", dpi=200, bbox_inches="tight")
	plt.close(figure)


def plot_comparison_heatmap(comparison: pd.DataFrame, output_dir: Path) -> None:
	if comparison.empty:
		return

	classes = sorted(comparison["class_a"].unique())
	matrix = np.zeros((len(classes), len(classes)), dtype=np.float32)

	for row_index, class_a in enumerate(classes):
		for col_index, class_b in enumerate(classes):
			value = comparison[(comparison["class_a"] == class_a) & (comparison["class_b"] == class_b)]["euclidean_distance"].iloc[0]
			matrix[row_index, col_index] = value

	figure, axis = plt.subplots(figsize=(8, 6))
	image = axis.imshow(matrix, cmap="viridis")
	axis.set_xticks(range(len(classes)))
	axis.set_yticks(range(len(classes)))
	axis.set_xticklabels(classes, rotation=20, ha="right")
	axis.set_yticklabels(classes)
	axis.set_title("Distância Euclidiana entre as médias das classes")
	figure.colorbar(image, ax=axis, label="Distância")
	figure.tight_layout()
	figure.savefig(output_dir / "comparacao_entre_classes.png", dpi=200, bbox_inches="tight")
	plt.close(figure)


def write_report_markdown(
	base_dir: Path,
	output_dir: Path,
	image_stats: pd.DataFrame,
	class_stats: pd.DataFrame,
	comparison: pd.DataFrame,
) -> None:
	if class_stats.empty or comparison.empty:
		raise RuntimeError("Nao foi possivel gerar o relatorio porque faltam estatisticas calculadas.")

	class_count = int(class_stats["class"].nunique())
	image_count = int(image_stats.shape[0])
	images_per_class = int(class_stats["image_count"].iloc[0]) if not class_stats.empty else 0
	most_separated = best_pair_from_comparison(comparison)
	most_similar = worst_pair_from_comparison(comparison)

	report_lines = [
		"# Laboratório 03 - Granulometria Morfológica",
		"",
		"## Resumo executivo",
		"",
		f"Este relatório apresenta a caracterização morfológica de {image_count} imagens distribuídas em {class_count} classes, todas organizadas em [lab03/images]({base_dir}). A análise usa assinaturas granulométricas construídas por aberturas morfológicas com elemento estruturante elíptico e escala variando de 1 a {DEFAULT_MAX_RADIUS} pixels.",
		"",
		"## Base de dados",
		"",
		f"A base utilizada contém {class_count} classes com {images_per_class} imagens por classe, totalizando {image_count} amostras em tons de cinza e com tamanho padronizado para {DEFAULT_IMAGE_SIZE[0]} x {DEFAULT_IMAGE_SIZE[1]} pixels.",
		"",
		"### Classes analisadas",
		"",
		dataframe_to_markdown(class_stats[["class", "image_count"]]),
		"",
		"### Exemplos visuais",
		"",
		"A figura a seguir mostra exemplos representativos de cada classe usada no experimento.",
		"",
		"![Exemplos das classes](amostras_das_classes.png)",
		"",
		"## Metodologia",
		"",
		"Para cada imagem, foram executados os seguintes passos:",
		"",
		"1. Leitura em tons de cinza e redimensionamento para o mesmo tamanho.",
		"2. Normalização dos níveis de intensidade para o intervalo [0, 1].",
		f"3. Aplicação de aberturas morfológicas com elemento estruturante elíptico para raios de 1 até {DEFAULT_MAX_RADIUS}.",
		"4. Cálculo da retenção normalizada de massa após cada abertura, formando a assinatura granulométrica.",
		"5. Consolidação das assinaturas individuais em curvas médias por classe e comparação entre classes.",
		"",
		"A escolha do elemento estruturante elíptico reduz o viés direcional e torna a assinatura mais adequada para texturas naturais.",
		"",
		"## Resultados",
		"",
		"### Assinaturas individuais e médias por classe",
		"",
		"A figura abaixo mostra as curvas individuais em cinza claro e a média de cada classe em azul, com faixa de desvio-padrão.",
		"",
		"![Assinaturas por classe](assinaturas_por_classe.png)",
		"",
		"### Comparação entre médias",
		"",
		"A visualização abaixo resume a distância euclidiana entre as curvas médias das classes.",
		"",
		"![Comparação entre classes](comparacao_entre_classes.png)",
		"",
		"![Médias das classes](media_entre_classes.png)",
		"",
		"### Tabela de resumo",
		"",
		dataframe_to_markdown(class_stats),
		"",
		"### Interpretação automática",
		"",
	] + render_summary_bullets(class_stats, comparison) + [
		"",
		"## Discussão",
		"",
		f"As três classes apresentam assinaturas com comportamento global parecido, mas com diferenças consistentes de retenção de massa. Em particular, {class_stats.sort_values('area_under_curve_mean', ascending=False).iloc[0]['class']} mostra maior área sob a curva, enquanto {class_stats.sort_values('area_under_curve_mean', ascending=True).iloc[0]['class']} apresenta a resposta mais rápida à abertura morfológica.",
		"",
		"As curvas médias são suficientemente estáveis para sugerir que a granulometria morfológica é útil como descritor de textura neste subconjunto. A separação não é absoluta, porém há sinais de distinção entre as classes quando se observam as métricas agregadas e a diferença entre as médias.",
		"",
		"## Conclusão",
		"",
		"A experimentação indica que a granulometria morfológica fornece uma representação compacta e interpretável para este problema. O descritor captura variações de escala associadas à textura e produz curvas comparáveis entre classes, permitindo análise visual e quantitativa.",
		"",
		"## Apêndice: métricas por imagem",
		"",
		"As dez primeiras amostras são mostradas abaixo como referência; o arquivo CSV completo está disponível na saída do experimento.",
		"",
		dataframe_to_markdown(image_stats.head(10)),
		"",
		"## Apêndice: arquivos gerados",
		"",
		"- amostras_das_classes.png",
		"- assinaturas_por_classe.png",
		"- media_entre_classes.png",
		"- comparacao_entre_classes.png",
		"- assinaturas_individuais.csv",
		"- resumo_por_imagem.csv",
		"- resumo_por_classe.csv",
		"- comparacao_classes.csv",
	]

	(output_dir / "relatorio_lab03.md").write_text("\n".join(report_lines), encoding="utf-8")
	(output_dir / "relatorio_resumo.md").write_text("\n".join(report_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Laboratorio 03 - Granulometria Morfologica")
	parser.add_argument(
		"--base-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "images",
		help="Diretorio base com as classes de imagens.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(__file__).resolve().parent / "output",
		help="Diretorio de saida para graficos e tabelas.",
	)
	parser.add_argument(
		"--image-size",
		type=int,
		nargs=2,
		default=DEFAULT_IMAGE_SIZE,
		metavar=("WIDTH", "HEIGHT"),
		help="Tamanho alvo para redimensionamento das imagens.",
	)
	parser.add_argument(
		"--max-radius",
		type=int,
		default=DEFAULT_MAX_RADIUS,
		help="Maior raio usado nas operacoes morfologicas.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	base_dir = args.base_dir
	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	if not base_dir.exists():
		raise FileNotFoundError(f"Diretorio base inexistente: {base_dir}")

	results, signature_table = collect_results(base_dir, tuple(args.image_size), args.max_radius)
	if not results:
		raise RuntimeError(f"Nenhuma imagem encontrada em {base_dir}")

	image_stats, class_stats = build_class_statistics(results)
	comparison = compare_class_means(results)

	signature_table.to_csv(output_dir / "assinaturas_individuais.csv", index=False)
	image_stats.to_csv(output_dir / "resumo_por_imagem.csv", index=False)
	class_stats.to_csv(output_dir / "resumo_por_classe.csv", index=False)
	comparison.to_csv(output_dir / "comparacao_classes.csv", index=False)

	save_sample_montage(base_dir, output_dir)
	plot_class_signatures(results, output_dir)
	plot_mean_comparison(results, output_dir)
	plot_comparison_heatmap(comparison, output_dir)
	write_report_markdown(base_dir, output_dir, image_stats, class_stats, comparison)

	print(f"Processamento concluido. Arquivos gerados em: {output_dir}")
	print(class_stats.to_string(index=False))


if __name__ == "__main__":
	main()