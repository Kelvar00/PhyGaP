<script>
import Title from './sections/Title.vue'
import Teaser from './sections/Teaser.vue'
import Abstract from './sections/Abstract.vue'
import Pipeline from './sections/pipeline.vue'
import Results from './sections/Results.vue'
import RelightingComparison from './sections/RelightingComparison.vue'
import BibTeX from './sections/BibTeX.vue'

export default {
  components: {
    Title,
    Teaser,
    Pipeline,
    Abstract,
    Results,
    RelightingComparison,
    BibTeX,
  }
}
</script>

<template>
  <Title/>
  
  <Pipeline/>
  <Abstract/>

  <Teaser/>
  <RelightingComparison/>
  <Results/>
  <BibTeX/>
</template>
