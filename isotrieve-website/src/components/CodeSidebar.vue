<template>
  <div class="code-sidebar">
    <div class="code-sidebar-header">
      <span class="code-sidebar-title">{{ title }}</span>
      <div class="code-lang-tabs">
        <button
          v-for="lang in languages"
          :key="lang"
          class="code-lang-tab"
          :class="{ active: activeLang === lang }"
          @click="activeLang = lang"
        >
          {{ lang }}
        </button>
      </div>
    </div>
    <div class="code-sidebar-content">
      <div
        v-for="lang in languages"
        :key="lang"
        class="code-sidebar-block"
        :class="{ active: activeLang === lang }"
        :data-lang="lang"
      >
        <pre><code>{{ codeBlocks[lang] }}</code></pre>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'CodeSidebar',
  props: {
    title: {
      type: String,
      default: 'Code Example'
    },
    codeBlocks: {
      type: Object,
      required: true
    }
  },
  data() {
    const languages = this.codeBlocks ? Object.keys(this.codeBlocks) : []
    return {
      activeLang: languages[0] || 'python'
    }
  },
  computed: {
    languages() {
      return this.codeBlocks ? Object.keys(this.codeBlocks) : []
    }
  }
}
</script>
